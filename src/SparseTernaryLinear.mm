#include "SparseTernaryLinear.h"
#include "Adafactor.h"
#include "MetalContext.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <numeric>

struct uint2_pod { uint32_t x, y; };

SparseTernaryLinear::SparseTernaryLinear(int in_features, int out_features, float density)
    : in_features(in_features), out_features(out_features), init_density(density)
{
    const std::string kpath = "kernels/ops.metal";
    sparseForwardKernel           = std::make_unique<Kernel>("sparse_ternary_matmul_free",    kpath);
    sparseBackwardInputKernel     = std::make_unique<Kernel>("sparse_ternary_backward_input", kpath);
    sparseBackwardWeightPosKernel = std::make_unique<Kernel>("sparse_backward_weight_pos",    kpath);
    sparseBackwardWeightNegKernel = std::make_unique<Kernel>("sparse_backward_weight_neg",    kpath);
    addKernel                     = std::make_unique<Kernel>("elementwise_add",               kpath);
    initialize_sparse_weights(density);
}

// BF16 ↔ float helpers (CPU)
static inline float bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16; float f; std::memcpy(&f, &b, 4); return f;
}
static inline uint16_t f32_to_bf16(float v) {
    uint32_t b; std::memcpy(&b, &v, 4);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

void SparseTernaryLinear::initialize_sparse_weights(float density) {
    static uint32_t init_counter = 0;
    std::default_random_engine gen(42 + (init_counter++));
    std::uniform_real_distribution<float> uni(0.f, 1.f);

    int expected_nnz = std::max(1, int(float(out_features) * float(in_features) * density * 0.5f));

    std::vector<uint32_t> all_pos, all_neg;
    all_pos.reserve(expected_nnz);
    all_neg.reserve(expected_nnz);

    std::vector<uint2_pod> all_ptrs(out_features), all_counts(out_features);
    std::vector<std::vector<uint32_t>> col_pos(in_features), col_neg(in_features);
    { int avg = std::max(1, int(float(out_features)*density*0.5f));
      for (auto& v:col_pos) v.reserve(avg);
      for (auto& v:col_neg) v.reserve(avg); }

    std::vector<uint32_t> flat_pos_row, flat_pos_col, flat_neg_row, flat_neg_col;
    flat_pos_row.reserve(expected_nnz); flat_pos_col.reserve(expected_nnz);
    flat_neg_row.reserve(expected_nnz); flat_neg_col.reserve(expected_nnz);

    // Standard Xavier init for sparse weights
    float scale = 1.f / sqrtf((float)in_features);
    std::vector<float> init_pos_w, init_neg_w;
    init_pos_w.reserve(expected_nnz);
    init_neg_w.reserve(expected_nnz);

    uint32_t wseed = 42u + init_counter;

    for (int i = 0; i < out_features; ++i) {
        all_ptrs[i].x = (uint32_t)all_pos.size();
        all_ptrs[i].y = (uint32_t)all_neg.size();
        uint32_t np = 0, nn = 0;
        for (int j = 0; j < in_features; ++j) {
            if (uni(gen) < density) {
                wseed = wseed * 1664525u + 1013904223u;
                float w = ((float)(wseed >> 1) / (float)0x7FFFFFFF - 1.f) * scale;
                if (uni(gen) < 0.5f) {
                    all_pos.push_back(j);
                    col_pos[j].push_back(i);
                    flat_pos_row.push_back(i); flat_pos_col.push_back(j);
                    init_pos_w.push_back(w);
                    ++np;
                } else {
                    all_neg.push_back(j);
                    col_neg[j].push_back(i);
                    flat_neg_row.push_back(i); flat_neg_col.push_back(j);
                    init_neg_w.push_back(w);
                    ++nn;
                }
            }
        }
        all_counts[i].x = np;
        all_counts[i].y = nn;
    }

    // CSC
    std::vector<uint32_t> all_pos_col, all_neg_col;
    std::vector<uint2_pod> all_ptrs_col(in_features), all_counts_col(in_features);
    for (int j = 0; j < in_features; ++j) {
        all_ptrs_col[j].x = (uint32_t)all_pos_col.size();
        all_ptrs_col[j].y = (uint32_t)all_neg_col.size();
        all_pos_col.insert(all_pos_col.end(), col_pos[j].begin(), col_pos[j].end());
        all_neg_col.insert(all_neg_col.end(), col_neg[j].begin(), col_neg[j].end());
        all_counts_col[j].x = (uint32_t)col_pos[j].size();
        all_counts_col[j].y = (uint32_t)col_neg[j].size();
    }

    _nnz_pos = (uint32_t)flat_pos_row.size();
    _nnz_neg = (uint32_t)flat_neg_row.size();

    std::cout << "  SparseTernaryLinear [" << out_features << "x" << in_features
              << "] nnz=" << (_nnz_pos + _nnz_neg)
              << " (" << std::fixed << std::setprecision(3)
              << float(_nnz_pos + _nnz_neg) / float(out_features * in_features)
              << " density)\n";

    auto mkU32 = [](size_t n) {
        return std::make_shared<Tensor>(std::vector<int>{(int)n}, DType::UInt32);
    };
    auto mkBF16 = [](size_t n) {
        return std::make_shared<Tensor>(std::vector<int>{(int)n}, DType::BFloat16);
    };

    pos_indices = mkU32(all_pos.size());
    neg_indices = mkU32(all_neg.size());
    row_ptrs    = std::make_shared<Tensor>(std::vector<int>{out_features, 2}, DType::UInt32);
    row_counts  = std::make_shared<Tensor>(std::vector<int>{out_features, 2}, DType::UInt32);
    pos_indices_col = mkU32(all_pos_col.size());
    neg_indices_col = mkU32(all_neg_col.size());
    col_ptrs    = std::make_shared<Tensor>(std::vector<int>{in_features, 2}, DType::UInt32);
    col_counts  = std::make_shared<Tensor>(std::vector<int>{in_features, 2}, DType::UInt32);
    pos_row_idx = mkU32(_nnz_pos);
    pos_col_idx = mkU32(_nnz_pos);
    neg_row_idx = mkU32(_nnz_neg);
    neg_col_idx = mkU32(_nnz_neg);

    // ── master_weights: NNZ 크기만 (dense full matrix 아님) ──────────
    // pos/neg 분리 저장 → 1.18GB 절약 (dim=2048, 24 layers)
    uint32_t np = std::max(1u, _nnz_pos), nn = std::max(1u, _nnz_neg);
    master_weights_pos = mkBF16(np);
    master_weights_neg = mkBF16(nn);

    // prefix-sum
    auto build_prefix = [&](const std::vector<uint2_pod>& c, bool x, int n) {
        std::vector<uint32_t> p(n+1, 0);
        for (int k=0;k<n;++k) p[k+1]=p[k]+(x?c[k].x:c[k].y);
        return p;
    };
    auto p_csr_pos = build_prefix(all_counts,     true,  out_features);
    auto p_csr_neg = build_prefix(all_counts,     false, out_features);
    auto p_csc_pos = build_prefix(all_counts_col, true,  in_features);
    auto p_csc_neg = build_prefix(all_counts_col, false, in_features);

    pos_csr_row_ptr = mkU32(out_features + 1);
    neg_csr_row_ptr = mkU32(out_features + 1);
    pos_csc_col_ptr = mkU32(in_features  + 1);
    neg_csc_col_ptr = mkU32(in_features  + 1);

    auto copy = [](std::shared_ptr<Tensor>& dst, const void* src) {
        if (dst->size() > 0 && src) std::memcpy(dst->data(), src, dst->bytes());
    };
    if (!all_pos.empty())     copy(pos_indices,     all_pos.data());
    if (!all_neg.empty())     copy(neg_indices,     all_neg.data());
    copy(row_ptrs,   all_ptrs.data());
    copy(row_counts, all_counts.data());
    if (!all_pos_col.empty()) copy(pos_indices_col, all_pos_col.data());
    if (!all_neg_col.empty()) copy(neg_indices_col, all_neg_col.data());
    copy(col_ptrs,   all_ptrs_col.data());
    copy(col_counts, all_counts_col.data());
    if (_nnz_pos > 0) { copy(pos_row_idx, flat_pos_row.data()); copy(pos_col_idx, flat_pos_col.data()); }
    if (_nnz_neg > 0) { copy(neg_row_idx, flat_neg_row.data()); copy(neg_col_idx, flat_neg_col.data()); }
    copy(pos_csr_row_ptr, p_csr_pos.data());
    copy(neg_csr_row_ptr, p_csr_neg.data());
    copy(pos_csc_col_ptr, p_csc_pos.data());
    copy(neg_csc_col_ptr, p_csc_neg.data());

    // master_weights_pos/neg 초기화 (NNZ 인덱스 순서대로)
    { uint16_t* p = (uint16_t*)master_weights_pos->data();
      for (size_t k=0;k<init_pos_w.size();++k) p[k] = f32_to_bf16(init_pos_w[k]); }
    { uint16_t* p = (uint16_t*)master_weights_neg->data();
      for (size_t k=0;k<init_neg_w.size();++k) p[k] = f32_to_bf16(init_neg_w[k]); }

    // packed weights
    packed_pos_w     = mkBF16(np);
    packed_neg_w     = mkBF16(nn);
    packed_pos_w_csc = mkBF16(np);
    packed_neg_w_csc = mkBF16(nn);
    sync_packed_weights();
}

void SparseTernaryLinear::sync_packed_weights() {
    // master_weights_pos[k] → packed_pos_w[k] (1:1, NNZ indexed)
    if (_nnz_pos > 0) {
        std::memcpy(packed_pos_w->data(),     master_weights_pos->data(), _nnz_pos * 2);
        std::memcpy(packed_pos_w_csc->data(), master_weights_pos->data(), _nnz_pos * 2);
    }
    if (_nnz_neg > 0) {
        std::memcpy(packed_neg_w->data(),     master_weights_neg->data(), _nnz_neg * 2);
        std::memcpy(packed_neg_w_csc->data(), master_weights_neg->data(), _nnz_neg * 2);
    }
}

// ── Forward ───────────────────────────────────────────────────────────────────
std::vector<std::shared_ptr<Tensor>> SparseTernaryLinear::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs)
{
    if (inputs.empty()) return {};
    auto input = inputs[0];
    last_input = input;

    auto in_shape = input->getShape();
    size_t total_items = 1;
    for (size_t i = 0; i+1 < in_shape.size(); ++i) total_items *= (size_t)in_shape[i];

    std::vector<int> out_shape = in_shape;
    out_shape.back() = out_features;
    auto output = std::make_shared<Tensor>(out_shape, DType::Float32);

    // output_scale = 1.0: scaling handled externally.
    // Body layers: RMS norm normalizes activation magnitudes between layers.
    // LM head: main.mm applies logit_scale after GPU commit.
    float output_scale = 1.0f;
    struct Params { uint32_t b, i, o; float scale; } p = {
        (uint32_t)total_items, (uint32_t)in_features, (uint32_t)out_features, output_scale };

    sparseForwardKernel->dispatch2D(
        {input.get(), pos_indices.get(), neg_indices.get(),
         row_ptrs.get(), row_counts.get(),
         packed_pos_w.get(), packed_neg_w.get()},
        {output.get()},
        out_features, (int)total_items, &p, sizeof(p));

    return {output};
}

// ── Backward ──────────────────────────────────────────────────────────────────
std::vector<std::shared_ptr<Tensor>> SparseTernaryLinear::backward(
    const std::vector<std::shared_ptr<Tensor>>& grad_outputs)
{
    if (grad_outputs.empty() || !last_input) return {};
    auto grad_out = grad_outputs[0];
    auto go_shape = grad_out->getShape();
    size_t total_items = 1;
    for (size_t i = 0; i+1 < go_shape.size(); ++i) total_items *= (size_t)go_shape[i];

    std::vector<int> gi_shape = go_shape;
    gi_shape.back() = in_features;
    auto grad_in = std::make_shared<Tensor>(gi_shape, DType::Float32);

    float output_scale = 1.0f;
    struct SP { uint32_t b, i, o; float scale; } sp = {
        (uint32_t)total_items, (uint32_t)in_features, (uint32_t)out_features, output_scale };

    // backward_input: grad_in = scale * W^T @ grad_out
    {
        auto enc = CommandBatch::get().encoder();
        auto& k = *sparseBackwardInputKernel;
        [enc setComputePipelineState:k.getPipelineState()];
        int idx = 0;
        [enc setBuffer:grad_in.get()->getBuffer()         offset:0 atIndex:idx++];
        [enc setBuffer:grad_out.get()->getBuffer()        offset:0 atIndex:idx++];
        [enc setBuffer:pos_indices_col.get()->getBuffer() offset:0 atIndex:idx++];
        [enc setBuffer:neg_indices_col.get()->getBuffer() offset:0 atIndex:idx++];
        [enc setBuffer:col_ptrs.get()->getBuffer()        offset:0 atIndex:idx++];
        [enc setBuffer:col_counts.get()->getBuffer()      offset:0 atIndex:idx++];
        [enc setBuffer:packed_pos_w_csc.get()->getBuffer()offset:0 atIndex:idx++];
        [enc setBuffer:packed_neg_w_csc.get()->getBuffer()offset:0 atIndex:idx++];
        [enc setBytes:&sp length:sizeof(sp) atIndex:idx++];
        constexpr uint32_t NNZ_TGS = 32;
        [enc dispatchThreads:MTLSizeMake(in_features, total_items * NNZ_TGS, 1)
         threadsPerThreadgroup:MTLSizeMake(1, NNZ_TGS, 1)];
    }

    // weight gradient: gw = scale * grad_out @ input^T
    struct WP { uint32_t batch_size, in_features, out_features, nnz; float scale; };
    
    std::shared_ptr<Tensor> gw_new_pos, gw_new_neg;
    if (_nnz_pos > 0) {
        gw_new_pos = std::make_shared<Tensor>(std::vector<int>{(int)_nnz_pos}, DType::Float32);
        WP wp{ (uint32_t)total_items, (uint32_t)in_features, (uint32_t)out_features, _nnz_pos, output_scale };
        sparseBackwardWeightPosKernel->dispatch(
            {grad_out.get(), last_input.get(), pos_row_idx.get(), pos_col_idx.get()},
            {gw_new_pos.get()}, &wp, sizeof(wp));
    }
    if (_nnz_neg > 0) {
        gw_new_neg = std::make_shared<Tensor>(std::vector<int>{(int)_nnz_neg}, DType::Float32);
        WP wp{ (uint32_t)total_items, (uint32_t)in_features, (uint32_t)out_features, _nnz_neg, output_scale };
        sparseBackwardWeightNegKernel->dispatch(
            {grad_out.get(), last_input.get(), neg_row_idx.get(), neg_col_idx.get()},
            {gw_new_neg.get()}, &wp, sizeof(wp));
    }

    // Single commit for both weight gradients + backward_input
    CommandBatch::get().commit_and_wait();

    // CPU accumulation (safe: GPU work is done)
    if (gw_new_pos) {
        if (gw_pos) {
            float* dst = (float*)gw_pos->data();
            float* src = (float*)gw_new_pos->data();
            for (uint32_t k = 0; k < _nnz_pos; ++k) dst[k] += src[k];
        } else {
            gw_pos = gw_new_pos;
        }
    }
    if (gw_new_neg) {
        if (gw_neg) {
            float* dst = (float*)gw_neg->data();
            float* src = (float*)gw_new_neg->data();
            for (uint32_t k = 0; k < _nnz_neg; ++k) dst[k] += src[k];
        } else {
            gw_neg = gw_new_neg;
        }
    }

    CommandBatch::get().begin();
    return {grad_in};
}

// ── Update (legacy, not called in fused path) ────────────────────────────────
void SparseTernaryLinear::update(Adafactor& /*optimizer*/) {
    // fused_adam_update가 대신 사용됨 (main.mm에서 직접 호출)
}

// ── Fused Adam update: scale + clip + Adam in one GPU pass ──────────────────
void SparseTernaryLinear::fused_adam_update(Adafactor& opt, float grad_scale,
                                            float clip_scale, int step_override)
{
    // Delegate to Adafactor's sparse fused step
    opt.step_sparse_fused(
        master_weights_pos, master_weights_neg,
        gw_pos, gw_neg,
        pos_row_idx, pos_col_idx,
        neg_row_idx, neg_col_idx,
        in_features,
        grad_scale, clip_scale,
        step_override);
}

void SparseTernaryLinear::clear_gradients() { gw_pos.reset(); gw_neg.reset(); }
void SparseTernaryLinear::clear_activations() { last_input.reset(); }

// ── Save/Load ─────────────────────────────────────────────────────────────────
void SparseTernaryLinear::save(std::ostream& os) const {
    os.write((const char*)&in_features,  sizeof(int));
    os.write((const char*)&out_features, sizeof(int));
    // NNZ 크기만 저장 (dense full matrix 아님)
    uint32_t np = _nnz_pos, nn = _nnz_neg;
    os.write((const char*)&np, sizeof(uint32_t));
    os.write((const char*)&nn, sizeof(uint32_t));
    if (np > 0) os.write((const char*)master_weights_pos->data(), np * 2);
    if (nn > 0) os.write((const char*)master_weights_neg->data(), nn * 2);
    // indices
    if (np > 0) {
        os.write((const char*)pos_indices->data(), pos_indices->bytes());
        os.write((const char*)pos_row_idx->data(), pos_row_idx->bytes());
        os.write((const char*)pos_col_idx->data(), pos_col_idx->bytes());
    }
    if (nn > 0) {
        os.write((const char*)neg_indices->data(), neg_indices->bytes());
        os.write((const char*)neg_row_idx->data(), neg_row_idx->bytes());
        os.write((const char*)neg_col_idx->data(), neg_col_idx->bytes());
    }
}

void SparseTernaryLinear::load(std::istream& is) {
    is.read((char*)&in_features,  sizeof(int));
    is.read((char*)&out_features, sizeof(int));
    uint32_t np, nn;
    is.read((char*)&np, sizeof(uint32_t));
    is.read((char*)&nn, sizeof(uint32_t));
    _nnz_pos = np; _nnz_neg = nn;
    // master_weights resize if needed
    if (!master_weights_pos || master_weights_pos->size() != np)
        master_weights_pos = std::make_shared<Tensor>(std::vector<int>{(int)std::max(1u,np)}, DType::BFloat16);
    if (!master_weights_neg || master_weights_neg->size() != nn)
        master_weights_neg = std::make_shared<Tensor>(std::vector<int>{(int)std::max(1u,nn)}, DType::BFloat16);
    if (np > 0) is.read((char*)master_weights_pos->data(), np * 2);
    if (nn > 0) is.read((char*)master_weights_neg->data(), nn * 2);
    if (np > 0) {
        is.read((char*)pos_indices->data(), pos_indices->bytes());
        is.read((char*)pos_row_idx->data(), pos_row_idx->bytes());
        is.read((char*)pos_col_idx->data(), pos_col_idx->bytes());
    }
    if (nn > 0) {
        is.read((char*)neg_indices->data(), neg_indices->bytes());
        is.read((char*)neg_row_idx->data(), neg_row_idx->bytes());
        is.read((char*)neg_col_idx->data(), neg_col_idx->bytes());
    }
    sync_packed_weights();
}

// ── Resparsify ────────────────────────────────────────────────────────────────
void SparseTernaryLinear::resparsify(float /*target_density*/) {
    // NNZ indexed master_weights 접근
    const uint16_t* mw_pos = (const uint16_t*)master_weights_pos->data();
    const uint16_t* mw_neg = (const uint16_t*)master_weights_neg->data();

    std::vector<float> active_abs;
    active_abs.reserve(_nnz_pos + _nnz_neg);
    for (uint32_t k=0;k<_nnz_pos;++k) active_abs.push_back(fabsf(bf16_to_f32(mw_pos[k])));
    for (uint32_t k=0;k<_nnz_neg;++k) active_abs.push_back(fabsf(bf16_to_f32(mw_neg[k])));

    if (active_abs.empty()) return;

    // 하위 10% weight를 0으로 설정 (pruning)
    size_t prune_count = active_abs.size() / 10;
    if (prune_count == 0) return;

    std::nth_element(active_abs.begin(), active_abs.begin() + prune_count, active_abs.end());
    float threshold = active_abs[prune_count];

    // 절대값이 threshold 이하인 weight를 0으로
    uint16_t zero_bf16 = f32_to_bf16(0.f);
    {
        uint16_t* wp = (uint16_t*)master_weights_pos->data();
        for (uint32_t k = 0; k < _nnz_pos; ++k) {
            if (fabsf(bf16_to_f32(wp[k])) <= threshold)
                wp[k] = zero_bf16;
        }
    }
    {
        uint16_t* wn = (uint16_t*)master_weights_neg->data();
        for (uint32_t k = 0; k < _nnz_neg; ++k) {
            if (fabsf(bf16_to_f32(wn[k])) <= threshold)
                wn[k] = zero_bf16;
        }
    }

    sync_packed_weights();
}