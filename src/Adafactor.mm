#include "Adafactor.h"
#include "SparseTernaryLinear.h"
#include "MetalContext.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

Adafactor::Adafactor(AdafactorParams p) : params(p) {
    const std::string kp = "kernels/ops.metal";
    k_reduce_row      = std::make_unique<Kernel>("adafactor_reduce",          kp);
    k_reduce_col      = std::make_unique<Kernel>("adafactor_reduce_col",      kp);
    k_reduce_row_bf16 = std::make_unique<Kernel>("adafactor_reduce_bf16",     kp);
    k_reduce_col_bf16 = std::make_unique<Kernel>("adafactor_reduce_col_bf16", kp);
    k_update          = std::make_unique<Kernel>("adafactor_update",          kp);
    k_update_bf16     = std::make_unique<Kernel>("adafactor_update_bf16",     kp);
    k_fill_bf16       = std::make_unique<Kernel>("fill_bf16",                 kp);
}

// Dense step
void Adafactor::step(std::shared_ptr<Tensor> weights, std::shared_ptr<Tensor> grads) {
    if (!weights || !grads) return;
    auto shape = weights->getShape();
    int out_f = shape[0], in_f = shape[1];
    Tensor* key = weights.get();

    if (dense_states.find(key) == dense_states.end()) {
        auto& s = dense_states[key];
        s.row_v = std::make_shared<Tensor>(std::vector<int>{out_f}, DType::BFloat16);
        s.col_v = std::make_shared<Tensor>(std::vector<int>{in_f},  DType::BFloat16);
        float zero = 0.f;
        k_fill_bf16->dispatch({}, {s.row_v.get()}, &zero, sizeof(float));
        k_fill_bf16->dispatch({}, {s.col_v.get()}, &zero, sizeof(float));
    }
    auto& s = dense_states[key];
    s.step_count++;

    struct GpuAdafactorParams { float lr,decay,e1,e2,clip; uint32_t b,i,o; int32_t step; };
    GpuAdafactorParams gp{ params.lr, params.decay_rate, params.epsilon1,
                           params.epsilon2, params.clip_threshold,
                           1, (uint32_t)in_f, (uint32_t)out_f, s.step_count };
    bool bf16w = (weights->getDType() == DType::BFloat16);
    auto& rr = bf16w ? *k_reduce_row_bf16 : *k_reduce_row;
    auto& rc = bf16w ? *k_reduce_col_bf16 : *k_reduce_col;
    rr.dispatch({grads.get()}, {s.row_v.get()}, &gp, sizeof(gp));
    rc.dispatch({grads.get()}, {s.col_v.get()}, &gp, sizeof(gp));
    CommandBatch::get().commit_and_wait();
    CommandBatch::get().begin();
    if (bf16w)
        k_update_bf16->dispatch2D({grads.get(), s.row_v.get(), s.col_v.get()},
                                  {weights.get()}, in_f, out_f, &gp, sizeof(gp));
    else
        k_update->dispatch2D({grads.get(), s.row_v.get(), s.col_v.get()},
                             {weights.get()}, in_f, out_f, &gp, sizeof(gp));
}

// 1-bit Adam sparse step
void Adafactor::step_sparse(
    std::shared_ptr<Tensor> weights,
    std::shared_ptr<Tensor> gw_pos, std::shared_ptr<Tensor> gw_neg,
    std::shared_ptr<Tensor> /*pos_row_idx*/, std::shared_ptr<Tensor> /*pos_col_idx*/,
    std::shared_ptr<Tensor> /*neg_row_idx*/, std::shared_ptr<Tensor> /*neg_col_idx*/,
    std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
    std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
    int /*out_f*/, int /*in_f*/)
{
    if (!weights) return;
    Tensor* key = weights.get();

    uint32_t nnz_pos = gw_pos ? (uint32_t)gw_pos->size() : 0u;
    uint32_t nnz_neg = gw_neg ? (uint32_t)gw_neg->size() : 0u;

    auto& s = sparse1bit_states[key];
    s.step_count++;

    if (s.m_pos.size() != nnz_pos) {
        s.m_pos.assign(nnz_pos, 0.f); s.v_pos.assign(nnz_pos, 0.f);
        s.m1_pos.assign((nnz_pos+31)/32, 0u); s.phase2 = false;
    }
    if (s.m_neg.size() != nnz_neg) {
        s.m_neg.assign(nnz_neg, 0.f); s.v_neg.assign(nnz_neg, 0.f);
        s.m1_neg.assign((nnz_neg+31)/32, 0u);
    }

    float beta1 = params.beta1;
    float beta2 = params.decay_rate;
    float eps   = params.epsilon2;
    float lr    = params.lr;
    (void)beta2;  // used implicitly in phase2 sign-based update

    if (!s.phase2 && s.step_count > params.warmup_steps)
        s.phase2 = true;

    auto bf16_to_f = [](uint16_t v) -> float {
        uint32_t b = (uint32_t)v << 16; float f; std::memcpy(&f,&b,4); return f; };
    auto f_to_bf16 = [](float v) -> uint16_t {
        uint32_t b; std::memcpy(&b,&v,4);
        b += 0x7FFFu + ((b>>16)&1u); return (uint16_t)(b>>16); };
    uint16_t* w = (uint16_t*)weights->data();

    auto update_side = [&](float* gf, uint32_t nnz,
                           std::vector<float>& m, std::vector<float>& v,
                           std::vector<uint32_t>& m1, float& m_scale) {
        if (nnz == 0 || !gf) return;
        // SGD with momentum: w -= lr*(beta1*m + g)
        if (!s.phase2) {
            for (uint32_t k = 0; k < nnz; ++k) {
                m[k] = beta1*m[k] + gf[k];
                float fw = bf16_to_f(w[k]);
                w[k] = f_to_bf16(fw - lr * m[k]);
            }
        } else {
            double sum_abs = 0.0;
            for (uint32_t k = 0; k < nnz; ++k) {
                m[k] = beta1*m[k] + (1.f-beta1)*gf[k];
                sum_abs += std::abs(m[k]);
            }
            m_scale = (nnz > 0) ? (float)(sum_abs/nnz) : 1e-8f;
            if (m_scale < 1e-10f) m_scale = 1e-10f;
            for (uint32_t k = 0; k < nnz; ++k) {
                if (m[k]>=0.f) m1[k/32]|=(1u<<(k%32)); else m1[k/32]&=~(1u<<(k%32));
            }
            for (uint32_t k = 0; k < nnz; ++k) {
                float sign = ((m1[k/32]>>(k%32))&1u) ? 1.f : -1.f;
                float step = lr * sign * m_scale / (std::sqrt(v[k]) + eps);
                float fw = bf16_to_f(w[k]);
                w[k] = f_to_bf16(fw - step);
            }
        }
    };

    float* gf_pos = gw_pos ? (float*)gw_pos->data() : nullptr;
    float* gf_neg = gw_neg ? (float*)gw_neg->data() : nullptr;
    update_side(gf_pos,nnz_pos,s.m_pos,s.v_pos,s.m1_pos,s.m_scale_pos);
    update_side(gf_neg,nnz_neg,s.m_neg,s.v_neg,s.m1_neg,s.m_scale_neg);
}

// Optimizer state save/load
static void write_vec_f(std::ostream& os, const std::vector<float>& v) {
    uint64_t n = v.size();
    os.write((const char*)&n, sizeof(n));
    if (n) os.write((const char*)v.data(), n * sizeof(float));
}
static void read_vec_f(std::istream& is, std::vector<float>& v) {
    uint64_t n; is.read((char*)&n, sizeof(n));
    v.resize(n);
    if (n) is.read((char*)v.data(), n * sizeof(float));
}

void Adafactor::save_state(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) { std::cerr << "[opt] cannot write " << path << "\n"; return; }
    const uint32_t magic = 0x4144414D;  // "ADAM"
    os.write((const char*)&magic, 4);

    // 유효한 state만 카운트
    int valid = 0;
    for (auto& [key, s] : sparse1bit_states)
        if (weight_to_idx.count(const_cast<Tensor*>(key))) valid++;
    printf("[opt-save] total states=%zu, registered=%zu, valid=%d\n",
           sparse1bit_states.size(), weight_to_idx.size(), valid);
    os.write((const char*)&valid, sizeof(int));

    for (auto& [key, s] : sparse1bit_states) {
        auto it = weight_to_idx.find(const_cast<Tensor*>(key));
        if (it == weight_to_idx.end()) continue;
        int idx = it->second;
        // 블록 크기 먼저 계산 (skip 시 사용)
        uint64_t block_bytes = sizeof(int)*2 + sizeof(bool)
            + sizeof(uint64_t)*4
            + (s.m_pos.size() + s.m_neg.size() + s.v_pos.size() + s.v_neg.size()) * sizeof(float);
        os.write((const char*)&block_bytes, sizeof(uint64_t));
        os.write((const char*)&idx, sizeof(int));
        os.write((const char*)&s.step_count, sizeof(int));
        os.write((const char*)&s.phase2, sizeof(bool));
        write_vec_f(os, s.m_pos); write_vec_f(os, s.m_neg);
        write_vec_f(os, s.v_pos); write_vec_f(os, s.v_neg);
    }
    std::cout << "[opt] saved " << valid << " layer states → " << path << "\n";
}

void Adafactor::load_state(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { std::cerr << "[opt] cannot read " << path << "\n"; return; }
    uint32_t magic; is.read((char*)&magic, 4);
    if (magic != 0x4144414D) { std::cerr << "[opt] bad magic\n"; return; }
    int n; is.read((char*)&n, sizeof(int));
    if (n < 0 || n > 10000) { std::cerr << "[opt] bad count " << n << "\n"; return; }
    int loaded = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t block_bytes; is.read((char*)&block_bytes, sizeof(uint64_t));
        if (!is || block_bytes > 2000000000ULL) {
            std::cerr << "[opt] corrupt block at " << i << "\n"; return;
        }
        auto block_start = is.tellg();
        int idx; is.read((char*)&idx, sizeof(int));
        auto it = idx_to_weight.find(idx);
        if (it == idx_to_weight.end()) {
            // 안전하게 seek로 skip
            is.seekg(block_start + (std::streamoff)block_bytes);
            continue;
        }
        Tensor* key = it->second;
        auto& s = sparse1bit_states[key];
        is.read((char*)&s.step_count, sizeof(int));
        is.read((char*)&s.phase2, sizeof(bool));
        read_vec_f(is, s.m_pos); read_vec_f(is, s.m_neg);
        read_vec_f(is, s.v_pos); read_vec_f(is, s.v_neg);
        s.m1_pos.assign((s.m_pos.size()+31)/32, 0u);
        for (size_t k=0;k<s.m_pos.size();++k)
            if(s.m_pos[k]>=0.f) s.m1_pos[k/32]|=(1u<<(k%32));
        s.m1_neg.assign((s.m_neg.size()+31)/32, 0u);
        for (size_t k=0;k<s.m_neg.size();++k)
            if(s.m_neg[k]>=0.f) s.m1_neg[k/32]|=(1u<<(k%32));
        loaded++;
    }
    std::cout << "[opt] loaded " << loaded << "/" << n << " layer states ← " << path << "\n";
}

// Fused GPU: scale + clip + Adam sparse update
void Adafactor::step_sparse_fused(
    std::shared_ptr<Tensor> weights_pos,
    std::shared_ptr<Tensor> weights_neg,
    std::shared_ptr<Tensor> gw_pos,
    std::shared_ptr<Tensor> gw_neg,
    std::shared_ptr<Tensor> /*pos_row_idx*/,
    std::shared_ptr<Tensor> /*pos_col_idx*/,
    std::shared_ptr<Tensor> /*neg_row_idx*/,
    std::shared_ptr<Tensor> /*neg_col_idx*/,
    int /*in_f*/,
    float grad_scale,
    float clip_scale,
    int step_count_override)
{
    if (!weights_pos) return;
    if (!k_fused_adam_sparse)
        k_fused_adam_sparse = std::make_unique<Kernel>(
            "fused_scale_adam_sparse_bf16", "kernels/ops.metal");

    // Use weights_pos as state key
    Tensor* key = weights_pos.get();
    auto& s = sparse1bit_states[key];
    s.step_count++;
    int sc = (step_count_override > 0) ? step_count_override : s.step_count;

    uint32_t nnz_pos = gw_pos ? (uint32_t)gw_pos->size() : 0u;
    uint32_t nnz_neg = gw_neg ? (uint32_t)gw_neg->size() : 0u;

    auto init_gpu_state = [&](uint32_t nnz,
                               std::vector<float>& m_cpu,
                               std::vector<float>& v_cpu,
                               std::shared_ptr<Tensor>& m_gpu,
                               std::shared_ptr<Tensor>& v_gpu) {
        if (!m_gpu || m_gpu->size() != nnz) {
            m_cpu.assign(nnz, 0.f); v_cpu.assign(nnz, 0.f);
            m_gpu = std::make_shared<Tensor>(std::vector<int>{(int)nnz}, DType::Float32);
            v_gpu = std::make_shared<Tensor>(std::vector<int>{(int)nnz}, DType::Float32);
            std::memset(m_gpu->data(), 0, m_gpu->bytes());
            std::memset(v_gpu->data(), 0, v_gpu->bytes());
        }
    };

    init_gpu_state(nnz_pos, s.m_pos, s.v_pos, s.m_pos_gpu, s.v_pos_gpu);
    init_gpu_state(nnz_neg, s.m_neg, s.v_neg, s.m_neg_gpu, s.v_neg_gpu);

    float beta1 = params.beta1;
    float beta2 = params.decay_rate;
    float bc1   = 1.f - std::pow(beta1, (float)sc);
    float bc2   = 1.f - std::pow(beta2, (float)sc);

    struct FusedAdamParams {
        float scale, clip_scale, lr, beta1, beta2, eps, bc1, bc2;
        uint32_t nnz, in_features;
    };

    // GPU Adam (NNZ-indexed)
    auto dispatch_side = [&](std::shared_ptr<Tensor> weights,
                              std::shared_ptr<Tensor> gw, uint32_t nnz,
                              std::shared_ptr<Tensor> m_gpu,
                              std::shared_ptr<Tensor> v_gpu) {
        if (nnz == 0 || !gw || !weights) return;
        FusedAdamParams fp{
            grad_scale, clip_scale,
            params.lr, beta1, beta2, params.epsilon2,
            bc1, bc2, nnz, 0u
        };
        k_fused_adam_sparse->dispatch2D(
            {gw.get(), m_gpu.get(), v_gpu.get()},
            {weights.get()},
            (int)nnz, 1,
            &fp, sizeof(fp));
    };
    dispatch_side(weights_pos, gw_pos, nnz_pos, s.m_pos_gpu, s.v_pos_gpu);
    dispatch_side(weights_neg, gw_neg, nnz_neg, s.m_neg_gpu, s.v_neg_gpu);
    CommandBatch::get().commit_and_wait();
    CommandBatch::get().begin();
}