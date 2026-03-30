#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <random>
#include <mach/mach.h>

#include "MetalContext.h"
#include "Kernel.h"
#include "Tensor.h"
#include "DataLoader.h"
#include "SparseAttention.h"
#include "SparseFFN.h"
#include "SparseTernaryLinear.h"
#include "Adafactor.h"
#include "TierManager.h"
#include "runtime/Config.h"

// 내가 리팩토링을 할거라 생각해..?

// 메모리 사용량 체크
static size_t get_gpu_mb() {
    id<MTLDevice> dev = MetalContext::getInstance().getDevice();
    return [dev currentAllocatedSize] / (1024 * 1024);
}
static size_t get_rss_mb() {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return info.resident_size / (1024 * 1024);
    return 0;
}

// BF16 helper
static inline uint16_t fp32_to_bf16(float v) {
    uint32_t b = *(uint32_t*)&v;
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

struct RMSNorm {
    float eps;
    std::unique_ptr<Kernel> fwd_k, bwd_k;

    RMSNorm() : eps(1e-6f) {}

    void init() {
        fwd_k = std::make_unique<Kernel>("rms_norm_forward",  "kernels/ops.metal");
        bwd_k = std::make_unique<Kernel>("rms_norm_backward", "kernels/ops.metal");
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        int rows = (int)(x->size() / x->getShape().back());
        uint32_t d = (uint32_t)x->getShape().back();
        auto out = std::make_shared<Tensor>(x->getShape(), DType::Float32);
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:fwd_k->getPipelineState()];
        [enc setBuffer:out->getBuffer() offset:0 atIndex:0];
        [enc setBuffer:x->getBuffer()   offset:0 atIndex:1];
        [enc setBytes:&d   length:4 atIndex:2];
        [enc setBytes:&eps length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(std::min(rows, 256), 1, 1)];
        return out;
    }

    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> dy,
                                     std::shared_ptr<Tensor> x_orig) {
        int rows = (int)(dy->size() / dy->getShape().back());
        uint32_t d = (uint32_t)dy->getShape().back();
        auto dx = std::make_shared<Tensor>(dy->getShape(), DType::Float32);
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:bwd_k->getPipelineState()];
        [enc setBuffer:dx->getBuffer()     offset:0 atIndex:0];
        [enc setBuffer:dy->getBuffer()     offset:0 atIndex:1];
        [enc setBuffer:x_orig->getBuffer() offset:0 atIndex:2];
        [enc setBytes:&d   length:4 atIndex:3];
        [enc setBytes:&eps length:4 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(std::min(rows, 256), 1, 1)];
        return dx;
    }
};

// Sparse Transformer Layer
struct SparseTransformerLayer {
    RMSNorm norm1, norm2;
    std::shared_ptr<SparseAttention> attn;
    std::shared_ptr<SparseFFN> ffn;

    std::shared_ptr<Tensor> r_pre_attn, r_pre_ffn;

    // HiMA: layer-level tier tracking
    int hima_base_idx = 0;  // first block index in TierManager

    void init(int dim, int num_heads, int block_size, float density, int ffn_dim) {
        norm1.init(); norm2.init();
        attn = std::make_shared<SparseAttention>(dim, num_heads, block_size, density);
        ffn  = std::make_shared<SparseFFN>(dim, density, ffn_dim);
    }

    std::vector<std::shared_ptr<SparseTernaryLinear>> all_sparse_weights() {
        auto aw = attn->get_internal_weights();
        auto fw = ffn->get_internal_weights();
        aw.insert(aw.end(), fw.begin(), fw.end());
        return aw;
    }

    // HiMA: layer의 dominant tier (가장 많은 블록의 tier)
    Tier dominant_tier(const TierManager& tm) const {
        int h=0, w=0, c=0;
        for (int i = 0; i < 7; ++i) {
            switch (tm.get_tier(hima_base_idx + i)) {
                case Tier::HOT:  h++; break;
                case Tier::WARM: w++; break;
                case Tier::COLD: c++; break;
            }
        }
        if (h >= w && h >= c) return Tier::HOT;
        if (w >= c) return Tier::WARM;
        return Tier::COLD;
    }

    void clear_saved() {
        r_pre_attn.reset(); r_pre_ffn.reset();
        attn->clear_activations();
        ffn->clear_activations();
    }
};

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static void mps_matmul(id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                       int M, int N, int K, bool transA = false, bool transB = false) {
    auto& ctx = MetalContext::getInstance();
    id<MTLCommandBuffer> cmd = [ctx.getCommandQueue() commandBuffer];
    int ldA = transA ? M : K;
    int ldB = transB ? K : N;
    MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:(transA?K:M) columns:(transA?M:K) rowBytes:ldA*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:(transB?N:K) columns:(transB?K:N) rowBytes:ldB*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*4 dataType:MPSDataTypeFloat32];
    MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:dA];
    MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:dB];
    MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C descriptor:dC];
    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
        initWithDevice:ctx.getDevice() transposeLeft:transA transposeRight:transB
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
    [gemm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    [cmd commit]; [cmd waitUntilCompleted];
}

struct FactoredEmbLMHead {
    int vocab, dim, k;  // k = factorization bottleneck

    // Sub-embedding: [vocab × k] — small, CPU lookup + GPU for LM head
    std::vector<float> sub_emb;         // CPU master
    std::vector<float> sub_emb_m, sub_emb_v;  // Adam states
    std::shared_ptr<Tensor> sub_emb_gpu;  // GPU copy for MPS

    // Embedding projection: [k × dim]
    std::vector<float> emb_proj;
    std::vector<float> emb_proj_m, emb_proj_v;
    std::shared_ptr<Tensor> emb_proj_gpu;

    // LM head projection: [dim × k]
    std::vector<float> lm_proj;
    std::vector<float> lm_proj_m, lm_proj_v;
    std::shared_ptr<Tensor> lm_proj_gpu;

    // Gradient accumulators
    std::vector<float> sub_emb_g;
    std::vector<int> sub_emb_g_count;
    std::shared_ptr<Tensor> emb_proj_g_gpu;
    std::shared_ptr<Tensor> lm_proj_g_gpu;
    std::shared_ptr<Tensor> sub_emb_lm_g_gpu;  // grad for sub_emb from LM head path

    // Saved for backward
    std::shared_ptr<Tensor> last_hidden, last_projected;

    std::unique_ptr<Kernel> add_kernel;
    int step_count = 0;

    void init(int vocab_, int dim_, int k_) {
        vocab = vocab_; dim = dim_; k = k_;

        // Sub-embedding [vocab × k]
        sub_emb.resize((size_t)vocab * k);
        sub_emb_m.assign((size_t)vocab * k, 0.f);
        sub_emb_v.assign((size_t)vocab * k, 0.f);
        sub_emb_g.assign((size_t)vocab * k, 0.f);
        sub_emb_g_count.assign(vocab, 0);
        sub_emb_gpu = std::make_shared<Tensor>(std::vector<int>{vocab, k}, DType::Float32);

        // Embedding projection [k × dim]
        emb_proj.resize((size_t)k * dim);
        emb_proj_m.assign((size_t)k * dim, 0.f);
        emb_proj_v.assign((size_t)k * dim, 0.f);
        emb_proj_gpu = std::make_shared<Tensor>(std::vector<int>{k, dim}, DType::Float32);
        emb_proj_g_gpu = std::make_shared<Tensor>(std::vector<int>{k, dim}, DType::Float32);

        // LM head projection [dim × k]
        lm_proj.resize((size_t)dim * k);
        lm_proj_m.assign((size_t)dim * k, 0.f);
        lm_proj_v.assign((size_t)dim * k, 0.f);
        lm_proj_gpu = std::make_shared<Tensor>(std::vector<int>{dim, k}, DType::Float32);
        lm_proj_g_gpu = std::make_shared<Tensor>(std::vector<int>{dim, k}, DType::Float32);

        sub_emb_lm_g_gpu = std::make_shared<Tensor>(std::vector<int>{k, vocab}, DType::Float32);

        // Xavier init
        auto xinit = [](std::vector<float>& w, int fan_in, int fan_out) {
            float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
            uint32_t seed = 42 + fan_in * 31 + fan_out * 17;
            for (auto& v : w) {
                seed = seed * 1664525u + 1013904223u;
                v = ((float)(seed >> 1) / (float)0x7FFFFFFF * 2.0f - 1.0f) * limit;
            }
        };
        xinit(sub_emb, vocab, k);
        xinit(emb_proj, k, dim);
        xinit(lm_proj, dim, k);

        sync_to_gpu();

        add_kernel = std::make_unique<Kernel>("elementwise_add", "kernels/ops.metal");

        std::memset(emb_proj_g_gpu->data(), 0, emb_proj_g_gpu->bytes());
        std::memset(lm_proj_g_gpu->data(), 0, lm_proj_g_gpu->bytes());
    }

    void sync_to_gpu() {
        std::memcpy(sub_emb_gpu->data(), sub_emb.data(), sub_emb.size() * 4);
        std::memcpy(emb_proj_gpu->data(), emb_proj.data(), emb_proj.size() * 4);
        std::memcpy(lm_proj_gpu->data(), lm_proj.data(), lm_proj.size() * 4);
    }
    // Step 1: CPU lookup sub_emb[tok] → [BS, k]
    // Step 2: MPS matmul [BS, k] × [k, dim] → [BS, dim]
    // Step 3: Add positional encoding
    // 이게 잘 되는건가..?
    std::shared_ptr<Tensor> emb_forward(
        std::shared_ptr<Tensor> tok, const std::vector<float>& pos_enc,
        int B, int S)
    {
        int BS = B * S;
        // Step 1: CPU sub-embedding lookup → [BS, k]
        auto sub_out = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
        float* sp = (float*)sub_out->data();
        int32_t* tp = (int32_t*)tok->data();
        for (int i = 0; i < BS; ++i) {
            int v = std::max(0, std::min(tp[i], vocab - 1));
            std::memcpy(sp + i * k, sub_emb.data() + (size_t)v * k, k * sizeof(float));
        }

        // Step 2: MPS [BS, k] × [k, dim] → [BS, dim]
        auto out = std::make_shared<Tensor>(std::vector<int>{BS, dim}, DType::Float32);
        mps_matmul(sub_out->getBuffer(), emb_proj_gpu->getBuffer(), out->getBuffer(),
                   BS, dim, k);

        // Step 3: Add positional encoding (CPU, fast for small dim)
        float* op = (float*)out->data();
        for (int i = 0; i < BS; ++i) {
            const float* pe = pos_enc.data() + (size_t)(i % S) * dim;
            for (int d = 0; d < dim; ++d)
                op[i * dim + d] += pe[d];
        }
        return out;
    }

    std::shared_ptr<Tensor> lm_forward(std::shared_ptr<Tensor> hidden, int BS) {
        last_hidden = hidden;

        // Commit any pending compute work
        CommandBatch::get().commit_and_wait();

        // Step 1: [BS, dim] × [dim, k] → [BS, k]
        last_projected = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
        mps_matmul(hidden->getBuffer(), lm_proj_gpu->getBuffer(), last_projected->getBuffer(),
                   BS, k, dim);

        // Step 2: [BS, k] × sub_emb^T → [BS, vocab]
        // sub_emb is [vocab, k], transposed = [k, vocab]
        auto logits = std::make_shared<Tensor>(std::vector<int>{BS, vocab}, DType::Float32);
        mps_matmul(last_projected->getBuffer(), sub_emb_gpu->getBuffer(), logits->getBuffer(),
                   BS, vocab, k, false, true);  // transB = true

        CommandBatch::get().begin();
        return logits;
    }

    // dlogits[BS, vocab] → dx[BS, dim]
    // Also accumulates gradients for lm_proj and sub_emb (from LM path)
    std::shared_ptr<Tensor> lm_backward(std::shared_ptr<Tensor> dlogits, int BS) {
        CommandBatch::get().commit_and_wait();

        // d_projected[BS, k] = dlogits[BS, vocab] × sub_emb[vocab, k]
        auto d_proj = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
        mps_matmul(dlogits->getBuffer(), sub_emb_gpu->getBuffer(), d_proj->getBuffer(),
                   BS, k, vocab);

        // dx[BS, dim] = d_projected[BS, k] × lm_proj^T[k, dim]
        auto dx = std::make_shared<Tensor>(std::vector<int>{BS, dim}, DType::Float32);
        mps_matmul(d_proj->getBuffer(), lm_proj_gpu->getBuffer(), dx->getBuffer(),
                   BS, dim, k, false, true);  // transB

        // g_lm_proj[dim, k] = hidden^T[dim, BS] × d_proj[BS, k]
        auto g_lm = std::make_shared<Tensor>(std::vector<int>{dim, k}, DType::Float32);
        mps_matmul(last_hidden->getBuffer(), d_proj->getBuffer(), g_lm->getBuffer(),
                   dim, k, BS, true, false);  // transA

        // g_sub_emb_lm[k, vocab] = projected^T[k, BS] × dlogits[BS, vocab]
        auto g_sub = std::make_shared<Tensor>(std::vector<int>{k, vocab}, DType::Float32);
        mps_matmul(last_projected->getBuffer(), dlogits->getBuffer(), g_sub->getBuffer(),
                   k, vocab, BS, true, false);  // transA

        // Accumulate gradients on GPU
        CommandBatch::get().begin();
        {
            uint32_t n;
            auto enc = CommandBatch::get().encoder();

            // lm_proj grad accumulate
            n = dim * k;
            [enc setComputePipelineState:add_kernel->getPipelineState()];
            [enc setBuffer:lm_proj_g_gpu->getBuffer() offset:0 atIndex:0];
            [enc setBuffer:g_lm->getBuffer()          offset:0 atIndex:1];
            [enc setBytes:&n length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        {
            auto enc = CommandBatch::get().encoder();
            uint32_t n = k * vocab;
            [enc setComputePipelineState:add_kernel->getPipelineState()];
            [enc setBuffer:sub_emb_lm_g_gpu->getBuffer() offset:0 atIndex:0];
            [enc setBuffer:g_sub->getBuffer()              offset:0 atIndex:1];
            [enc setBytes:&n length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }

        last_hidden.reset();
        last_projected.reset();
        return dx;
    }

    // ── Embedding backward: accumulate into sub_emb_g (CPU) ──
    // cg is [BS, dim], we need gradient through emb_proj and sub_emb
    void emb_backward(std::shared_ptr<Tensor> cg, std::shared_ptr<Tensor> tok, int BS) {
        // g_through_proj[BS, k] = cg[BS, dim] × emb_proj^T[dim, k]
        CommandBatch::get().commit_and_wait();
        auto g_k = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
        mps_matmul(cg->getBuffer(), emb_proj_gpu->getBuffer(), g_k->getBuffer(),
                   BS, k, dim, false, true);

        // emb_proj grad: emb_proj_g[k, dim] += sub_lookup^T[k, BS] × cg[BS, dim]
        // Need sub_lookup for this batch — reconstruct from tokens
        auto sub_out = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
        float* sp = (float*)sub_out->data();
        int32_t* tp = (int32_t*)tok->data();
        for (int i = 0; i < BS; ++i) {
            int v = std::max(0, std::min(tp[i], vocab - 1));
            std::memcpy(sp + i * k, sub_emb.data() + (size_t)v * k, k * sizeof(float));
        }

        auto g_ep = std::make_shared<Tensor>(std::vector<int>{k, dim}, DType::Float32);
        mps_matmul(sub_out->getBuffer(), cg->getBuffer(), g_ep->getBuffer(),
                   k, dim, BS, true, false);

        CommandBatch::get().begin();
        {
            uint32_t n = k * dim;
            auto enc = CommandBatch::get().encoder();
            [enc setComputePipelineState:add_kernel->getPipelineState()];
            [enc setBuffer:emb_proj_g_gpu->getBuffer() offset:0 atIndex:0];
            [enc setBuffer:g_ep->getBuffer()           offset:0 atIndex:1];
            [enc setBytes:&n length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        CommandBatch::get().commit_and_wait();

        // sub_emb grad: scatter g_k[BS, k] into sub_emb_g[vocab, k]
        float* gk = (float*)g_k->data();
        for (int i = 0; i < BS; ++i) {
            int v = std::max(0, std::min(tp[i], vocab - 1));
            float* dst = sub_emb_g.data() + (size_t)v * k;
            float* src = gk + (size_t)i * k;
            for (int d = 0; d < k; ++d) dst[d] += src[d];
            sub_emb_g_count[v]++;
        }

        CommandBatch::get().begin();
    }

    // ── Adam update for all factored parameters ──
    void adam_step(float lr, float accum_scale) {
        step_count++;
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float bc1 = 1.0f - powf(b1, (float)step_count);
        float bc2 = 1.0f - powf(b2, (float)step_count);

        // 1. sub_emb: CPU Adam (sparse update, only touched tokens)
        // Merge LM head gradient from GPU
        CommandBatch::get().commit_and_wait();
        float* lm_g = (float*)sub_emb_lm_g_gpu->data();
        // lm_g is [k, vocab] but sub_emb is [vocab, k] — transpose needed
        for (int v = 0; v < vocab; ++v) {
            float* e  = sub_emb.data()   + (size_t)v * k;
            float* mp = sub_emb_m.data() + (size_t)v * k;
            float* vp = sub_emb_v.data() + (size_t)v * k;
            float* ga = sub_emb_g.data() + (size_t)v * k;
            bool has_emb_g = (sub_emb_g_count[v] > 0);
            // Check if LM grad is nonzero for this vocab entry
            bool has_lm_g = false;
            for (int d = 0; d < k && !has_lm_g; ++d)
                if (lm_g[d * vocab + v] != 0.f) has_lm_g = true;
            if (!has_emb_g && !has_lm_g) continue;

            for (int d = 0; d < k; ++d) {
                float g = ga[d] * accum_scale + lm_g[d * vocab + v] * accum_scale;
                mp[d] = b1 * mp[d] + (1.f - b1) * g;
                vp[d] = b2 * vp[d] + (1.f - b2) * g * g;
                e[d] -= lr * (mp[d] / bc1) / (sqrtf(vp[d] / bc2) + eps);
                ga[d] = 0.f;
            }
            sub_emb_g_count[v] = 0;
        }
        std::memset(sub_emb_lm_g_gpu->data(), 0, sub_emb_lm_g_gpu->bytes());

        // 2. emb_proj: CPU Adam (small: k×dim)
        {
            float* ga = (float*)emb_proj_g_gpu->data();
            for (int i = 0; i < k * dim; ++i) {
                float g = ga[i] * accum_scale;
                emb_proj_m[i] = b1 * emb_proj_m[i] + (1.f - b1) * g;
                emb_proj_v[i] = b2 * emb_proj_v[i] + (1.f - b2) * g * g;
                emb_proj[i] -= lr * (emb_proj_m[i] / bc1) / (sqrtf(emb_proj_v[i] / bc2) + eps);
                ga[i] = 0.f;
            }
        }

        // 3. lm_proj: CPU Adam (small: dim×k)
        {
            float* ga = (float*)lm_proj_g_gpu->data();
            for (int i = 0; i < dim * k; ++i) {
                float g = ga[i] * accum_scale;
                lm_proj_m[i] = b1 * lm_proj_m[i] + (1.f - b1) * g;
                lm_proj_v[i] = b2 * lm_proj_v[i] + (1.f - b2) * g * g;
                lm_proj[i] -= lr * (lm_proj_m[i] / bc1) / (sqrtf(lm_proj_v[i] / bc2) + eps);
                ga[i] = 0.f;
            }
        }

        sync_to_gpu();
        CommandBatch::get().begin();
    }

    // ── Save/Load ──
    void save(std::ostream& f) const {
        f.write((const char*)&k, sizeof(int));
        f.write((const char*)sub_emb.data(), sub_emb.size() * 4);
        f.write((const char*)emb_proj.data(), emb_proj.size() * 4);
        f.write((const char*)lm_proj.data(), lm_proj.size() * 4);
    }
    void load(std::istream& f) {
        int k_file;
        f.read((char*)&k_file, sizeof(int));
        if (k_file != k) { std::cerr << "k mismatch: " << k_file << " vs " << k << "\n"; return; }
        f.read((char*)sub_emb.data(), sub_emb.size() * 4);
        f.read((char*)emb_proj.data(), emb_proj.size() * 4);
        f.read((char*)lm_proj.data(), lm_proj.size() * 4);
        sync_to_gpu();
    }
};

// helper
static std::vector<float> make_sinusoidal(int seq, int dim) {
    std::vector<float> pe((size_t)seq * dim, 0.f);
    for (int pos = 0; pos < seq; ++pos)
        for (int i = 0; i < dim; i += 2) {
            float f = 1.f / powf(10000.f, (float)i / dim);
            pe[(size_t)pos * dim + i] = sinf(pos * f);
            if (i + 1 < dim) pe[(size_t)pos * dim + i + 1] = cosf(pos * f);
        }
    return pe;
}

static void gpu_add_inplace(Kernel& add_k, std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b) {
    uint32_t n = (uint32_t)a->size();
    auto enc = CommandBatch::get().encoder();
    [enc setComputePipelineState:add_k.getPipelineState()];
    [enc setBuffer:a->getBuffer() offset:0 atIndex:0];
    [enc setBuffer:b->getBuffer() offset:0 atIndex:1];
    [enc setBytes:&n length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

// Tensor clone on unified memory — requires GPU work to be committed first
static std::shared_ptr<Tensor> tensor_clone(std::shared_ptr<Tensor> src) {
    auto dst = std::make_shared<Tensor>(src->getShape(), DType::Float32);
    std::memcpy(dst->data(), src->data(), src->bytes());
    return dst;
}

static float cosine_lr(float base_lr, int step, int warmup, int total) {
    if (step < warmup) return base_lr * (float)step / (float)warmup;
    float progress = (float)(step - warmup) / (float)std::max(total - warmup, 1);
    return base_lr * 0.5f * (1.0f + cosf((float)M_PI * progress));
}

// save , load
static void save_full_model(
    const std::string& path,
    const FactoredEmbLMHead& emb_lm,
    const std::vector<SparseTransformerLayer>& tl,
    int dim, int layers, int heads, int vocab, float density)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot save to " << path << "\n"; return; }

    uint32_t magic = 0x4A4C5832; // "JLX2" — factored format
    f.write((const char*)&magic, 4);
    f.write((const char*)&dim, sizeof(int));
    f.write((const char*)&layers, sizeof(int));
    f.write((const char*)&heads, sizeof(int));
    f.write((const char*)&vocab, sizeof(int));
    f.write((const char*)&density, sizeof(float));

    // Factored embedding + LM head
    emb_lm.save(f);

    // Sparse layers
    for (int i = 0; i < layers; ++i) {
        tl[i].attn->save(f);
        tl[i].ffn->save(f);
    }

    std::cout << "  [Save] Full model → " << path << " ("
              << f.tellp() / (1024*1024) << " MB)\n";
}

__attribute__((unused))
static bool load_full_model(
    const std::string& path,
    FactoredEmbLMHead& emb_lm,
    std::vector<SparseTransformerLayer>& tl,
    int dim, int layers, int heads, int vocab, float& density)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot load " << path << "\n"; return false; }

    uint32_t magic;
    f.read((char*)&magic, 4);
    if (magic != 0x4A4C5832) { std::cerr << "Bad magic (need JLX2)\n"; return false; }

    int d, l, h, v;
    f.read((char*)&d, sizeof(int));
    f.read((char*)&l, sizeof(int));
    f.read((char*)&h, sizeof(int));
    f.read((char*)&v, sizeof(int));
    f.read((char*)&density, sizeof(float));
    if (d != dim || l != layers || h != heads || v != vocab) {
        std::cerr << "Model mismatch\n";
        return false;
    }

    emb_lm.load(f);

    for (int i = 0; i < layers; ++i) {
        tl[i].attn->load(f);
        tl[i].ffn->load(f);
    }

    for (int i = 0; i < layers; ++i) {
        for (auto& w : tl[i].all_sparse_weights())
            w->sync_packed_weights();
    }

    std::cout << "  [Load] Full model ← " << path << "\n";
    return true;
}

int main() {
    Config& cfg = Config::getInstance();
    cfg.load("config.txt");
    cfg.print();

    const int dim     = cfg.getInt("DIM", 768);
    const int heads   = cfg.getInt("NUM_HEADS", 12);
    const int layers  = cfg.getInt("NUM_LAYERS", 6);
    const int seq     = cfg.getInt("SEQ_LEN", 128);
    const int batch   = cfg.getInt("BATCH_SIZE", 2);
    const float lr    = cfg.getFloat("LEARNING_RATE", 0.0003f);
    const int maxs    = cfg.getInt("MAX_STEPS", 100000);
    const int logi    = cfg.getInt("LOG_INTERVAL", 10);
    const int accum   = cfg.getInt("GRAD_ACCUM", 8);
    const bool overfit_test = cfg.getBool("OVERFIT_TEST", false);
    const std::string ddir = cfg.getString("DATASET_DIR", "data");
    const int block_size   = cfg.getInt("BLOCK_SIZE", 32);

    const int vocab    = 50257;
    const int BS       = batch * seq;
    const int ffn_dim  = dim * 4;
    const int warmup   = 50;
    const float accum_scale = 1.0f / (float)accum;

    // ── Sparse density ──
    const float density = cfg.getFloat("DENSITY", 0.05f);

    std::cout << "\n══════════════════════════════════════════════\n"
              << "  Sparse Ternary Transformer\n"
              << "  dim=" << dim << " heads=" << heads
              << " ffn=" << ffn_dim << "\n"
              << "  seq=" << seq << " batch=" << batch << " BS=" << BS << "\n"
              << "  grad_accum=" << accum
              << " effective_BS=" << BS * accum << "\n"
              << "  density=" << density
              << " (NNZ/matrix ≈ " << (int)(dim * dim * density) << ")\n"
              << "  lr=" << lr << " warmup=" << warmup << " cosine_decay\n"
              << (overfit_test ? "  *** OVERFIT TEST ***\n" : "")
              << "══════════════════════════════════════════════\n\n";

    auto loader = std::make_unique<DataLoader>(ddir, "shard_*.bin", batch, seq);

    std::shared_ptr<Tensor> cached_ti, cached_tt;
    if (overfit_test) {
        auto [ti0, tt0] = loader->get_batch();
        cached_ti = std::make_shared<Tensor>(ti0->getShape(), DType::Int32);
        cached_tt = std::make_shared<Tensor>(tt0->getShape(), DType::Int32);
        std::memcpy(cached_ti->data(), ti0->data(), ti0->bytes());
        std::memcpy(cached_tt->data(), tt0->data(), tt0->bytes());
    }

    // ── Factored Embedding + LM Head ──
    const int emb_k = cfg.getInt("EMB_K", 256);
    FactoredEmbLMHead emb_lm;
    emb_lm.init(vocab, dim, emb_k);
    auto pos_enc = make_sinusoidal(seq, dim);
    std::cout << "  Factored Emb+LM: vocab=" << vocab << " k=" << emb_k
              << " dim=" << dim << " (params: " << (vocab*emb_k + emb_k*dim + dim*emb_k)/1e6
              << "M vs " << 2.0*vocab*dim/1e6 << "M unfactored)\n";

    // ── Sparse Transformer layers ──
    std::vector<SparseTransformerLayer> tl(layers);
    for (int i = 0; i < layers; ++i) {
        std::cout << "  Layer " << i << ": Sparse Ternary (density=" << density << ")\n";
        tl[i].init(dim, heads, block_size, density, ffn_dim);
    }

    RMSNorm final_norm; final_norm.init();

    // ── Adafactor for sparse layers ──
    AdafactorParams opt_params;
    opt_params.lr = lr;
    opt_params.beta1 = 0.9f;
    opt_params.decay_rate = 0.999f;
    opt_params.epsilon2 = 1e-8f;
    opt_params.warmup_steps = warmup;
    Adafactor optimizer(opt_params);

    int weight_idx = 0;
    std::vector<std::shared_ptr<SparseTernaryLinear>> all_sparse_layers;
    for (int i = 0; i < layers; ++i) {
        tl[i].hima_base_idx = weight_idx;
        auto weights = tl[i].all_sparse_weights();
        for (auto& w : weights) {
            optimizer.register_weight(w->get_master_weights_pos().get(), weight_idx++);
            all_sparse_layers.push_back(w);
        }
    }
    std::cout << "  Registered " << all_sparse_layers.size()
              << " sparse weight matrices\n";

    // ── HiMA: Hierarchical Memory-Aware Training ──
    HiMAConfig hima_cfg;
    hima_cfg.hot_ratio  = 0.40f;
    hima_cfg.warm_ratio = 0.35f;
    hima_cfg.warm_update_interval = 10;
    hima_cfg.rebalance_interval   = 50;
    hima_cfg.grad_ema_alpha       = 0.1f;
    hima_cfg.min_steps_in_tier    = 20;
    hima_cfg.warm_lr_scale        = 0.1f;
    TierManager hima((int)all_sparse_layers.size(), hima_cfg);
    std::cout << "\n";

    // ── Shared Kernels ──
    Kernel ce_k("cross_entropy_loss", "kernels/ops.metal");
    Kernel add_k("elementwise_add", "kernels/ops.metal");

    std::cout << std::left
              << std::setw(7) << "Step" << std::setw(12) << "Loss"
              << std::setw(10) << "PPL" << std::setw(10) << "ms"
              << std::setw(10) << "Tok/s" << std::setw(6) << "GPU"
              << std::setw(6) << "RSS"
              << "Tokens\n" << std::string(70, '-') << "\n";
    std::cout.flush();

    double sms = -1;
    size_t total = 0;
    float best = 999.f;
    int opt_step = 0;

    for (int step = 1; step <= maxs; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        float loss_sum = 0;

        opt_step++;
        float cur_lr = cosine_lr(lr, opt_step, warmup, maxs);
        optimizer.set_lr(cur_lr);

        // Gradient Accumulation
        for (int micro = 0; micro < accum; ++micro) { @autoreleasepool {
            std::shared_ptr<Tensor> ti, tt;
            if (overfit_test) {
                ti = std::make_shared<Tensor>(cached_ti->getShape(), DType::Int32);
                tt = std::make_shared<Tensor>(cached_tt->getShape(), DType::Int32);
                std::memcpy(ti->data(), cached_ti->data(), cached_ti->bytes());
                std::memcpy(tt->data(), cached_tt->data(), cached_tt->bytes());
            } else {
                auto batch_pair = loader->get_batch();
                ti = batch_pair.first;
                tt = batch_pair.second;
            }
            ti->reshape({BS}); tt->reshape({BS});

            // ── FORWARD ──
            auto r = emb_lm.emb_forward(ti, pos_enc, batch, seq);

            for (int li = 0; li < layers; ++li) {
                auto& L = tl[li];

                L.r_pre_attn = tensor_clone(r);

                CommandBatch::get().begin();
                auto xn = L.norm1.forward(r);
                xn->reshape({batch, seq, dim});
                auto attn_out = L.attn->forward({xn})[0];
                attn_out->reshape({BS, dim});
                gpu_add_inplace(add_k, r, attn_out);
                CommandBatch::get().commit_and_wait();

                L.r_pre_ffn = tensor_clone(r);

                CommandBatch::get().begin();
                auto xn2 = L.norm2.forward(r);
                xn2->reshape({batch, seq, dim});
                auto ffn_out = L.ffn->forward({xn2})[0];
                ffn_out->reshape({BS, dim});
                gpu_add_inplace(add_k, r, ffn_out);
                CommandBatch::get().commit_and_wait();
            }

            auto r_pre_final = tensor_clone(r);
            CommandBatch::get().begin();
            auto xf = final_norm.forward(r);
            auto logits = emb_lm.lm_forward(xf, BS);

            // CE Loss
            auto lbuf = std::make_shared<Tensor>(
                std::vector<int>{BS}, DType::Float32);
            auto glog = std::make_shared<Tensor>(
                std::vector<int>{BS, vocab}, DType::Float32);
            CommandBatch::get().begin();
            { auto enc = CommandBatch::get().encoder();
              [enc setComputePipelineState:ce_k.getPipelineState()];
              [enc setBuffer:lbuf->getBuffer()   offset:0 atIndex:0];
              [enc setBuffer:glog->getBuffer()   offset:0 atIndex:1];
              [enc setBuffer:logits->getBuffer() offset:0 atIndex:2];
              [enc setBuffer:tt->getBuffer()     offset:0 atIndex:3];
              uint32_t b32 = BS, v32 = vocab;
              [enc setBytes:&b32 length:4 atIndex:4];
              [enc setBytes:&v32 length:4 atIndex:5];
              [enc dispatchThreadgroups:MTLSizeMake(BS, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)]; }
            CommandBatch::get().commit_and_wait();

            { float* lp = (float*)lbuf->data();
              for (int i = 0; i < BS; ++i)
                  loss_sum += std::isfinite(lp[i]) ? lp[i] : 20.f; }

            // ── BACKWARD ──
            // LM head + final norm
            auto cg = emb_lm.lm_backward(glog, BS);
            CommandBatch::get().begin();
            cg = final_norm.backward(cg, r_pre_final);
            CommandBatch::get().commit_and_wait();

            for (int li = layers - 1; li >= 0; --li) {
                auto& L = tl[li];
                Tier layer_tier = L.dominant_tier(hima);
                bool compute_weight_grad = (layer_tier != Tier::COLD);

                CommandBatch::get().begin();
                cg->reshape({batch, seq, dim});
                auto d_ffn = L.ffn->backward({cg})[0];
                d_ffn->reshape({BS, dim});
                auto d_norm2 = L.norm2.backward(d_ffn, L.r_pre_ffn);
                cg->reshape({BS, dim});
                gpu_add_inplace(add_k, cg, d_norm2);
                cg->reshape({batch, seq, dim});
                auto d_attn = L.attn->backward({cg})[0];
                d_attn->reshape({BS, dim});
                auto d_norm1 = L.norm1.backward(d_attn, L.r_pre_attn);
                cg->reshape({BS, dim});
                gpu_add_inplace(add_k, cg, d_norm1);
                CommandBatch::get().commit_and_wait();

                if (!compute_weight_grad) {
                    auto cold_weights = L.all_sparse_weights();
                    for (auto& w : cold_weights) w->clear_gradients();
                }

                L.clear_saved();
            }

            // Embedding gradients (factored)
            emb_lm.emb_backward(cg, ti, BS);
        }} // end micro-batch

        hima.step();

        // 1. Record gradient magnitudes for tier scoring
        for (int bi = 0; bi < (int)all_sparse_layers.size(); ++bi) {
            auto& w = all_sparse_layers[bi];
            float gnorm = 0.0f;
            int count = 0;
            auto gp = w->pos_gradients();
            auto gn = w->neg_gradients();
            if (gp) {
                float* gd = (float*)gp->data();
                int stride = std::max(1u, w->nnz_pos() / 256u);
                for (uint32_t k = 0; k < w->nnz_pos(); k += stride)
                    { gnorm += gd[k] * gd[k]; count++; }
            }
            if (gn) {
                float* gd = (float*)gn->data();
                int stride = std::max(1u, w->nnz_neg() / 256u);
                for (uint32_t k = 0; k < w->nnz_neg(); k += stride)
                    { gnorm += gd[k] * gd[k]; count++; }
            }
            if (count > 0) gnorm = sqrtf(gnorm / (float)count);
            hima.record_gradient(bi, gnorm);
        }

        // 2. Global grad norm for clipping (Hot + Warm only)
        float grad_norm_sq = 0.0f;
        for (int bi = 0; bi < (int)all_sparse_layers.size(); ++bi) {
            if (!hima.should_update(bi, opt_step)) continue;
            auto& w = all_sparse_layers[bi];
            auto gp = w->pos_gradients();
            auto gn = w->neg_gradients();
            if (gp) {
                float* gd = (float*)gp->data();
                for (uint32_t k = 0; k < w->nnz_pos(); ++k)
                    grad_norm_sq += gd[k] * gd[k];
            }
            if (gn) {
                float* gd = (float*)gn->data();
                for (uint32_t k = 0; k < w->nnz_neg(); ++k)
                    grad_norm_sq += gd[k] * gd[k];
            }
        }
        float grad_norm = sqrtf(grad_norm_sq);
        float clip_scale = (grad_norm > 1.0f && grad_norm > 0.f)
                           ? 1.0f / grad_norm : 1.0f;

        // 3. Tier-based optimizer step
        CommandBatch::get().begin();
        for (int bi = 0; bi < (int)all_sparse_layers.size(); ++bi) {
            if (!hima.should_update(bi, opt_step)) continue;

            float lr_mul = hima.lr_scale(bi);
            float orig_lr = optimizer.get_lr();
            optimizer.set_lr(orig_lr * lr_mul);

            all_sparse_layers[bi]->fused_adam_update(
                optimizer, accum_scale, clip_scale, opt_step);

            optimizer.set_lr(orig_lr);
        }
        CommandBatch::get().commit_and_wait();

        // Sync packed weights (only for updated blocks)
        for (int bi = 0; bi < (int)all_sparse_layers.size(); ++bi) {
            if (hima.should_update(bi, opt_step))
                all_sparse_layers[bi]->sync_packed_weights();
        }

        // Clear gradients for all (Cold already cleared in backward)
        for (auto& w : all_sparse_layers)
            w->clear_gradients();


        // 4. Rebalance tiers every R steps (after warmup)
        if (opt_step > warmup && opt_step % hima_cfg.rebalance_interval == 0) {
            int moves = hima.rebalance(opt_step);
            if (moves > 0 || opt_step % 200 == 0) {
                hima.print_status();
            }
        }

        // Factored Emb+LM head Adam
        emb_lm.adam_step(cur_lr, accum_scale);

        // Resparsify every 500 steps
        if (opt_step > 0 && opt_step % 500 == 0) {
            for (auto& w : all_sparse_layers) w->resparsify();
            std::cout << "  [Resparsify] step=" << opt_step << "\n";
        }

        float loss = loss_sum / (float)(BS * accum);
        if (!std::isfinite(loss)) continue;
        if (loss < best) best = loss;
        total += (size_t)BS * accum;

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (sms < 0) sms = ms; else sms = 0.95 * sms + 0.05 * ms;

        if (step % logi == 0 || step <= 5) {
            double ppl = exp(std::min((double)loss, 20.0));
            double toks = (double)(BS * accum) / (sms * 1e-3);
            std::cout << std::left
                      << std::setw(7) << step
                      << std::fixed << std::setprecision(4) << std::setw(12) << loss
                      << std::setprecision(1) << std::setw(10) << ppl
                      << std::setprecision(0) << std::setw(10) << ms
                      << std::setprecision(0) << std::setw(10) << toks
                      << std::setw(6) << get_gpu_mb()
                      << std::setw(6) << get_rss_mb()
                      << total << "\n";
            std::cout.flush();
        }
        if (step % 200 == 0) {
            std::cout << "  >>> step=" << step << " loss=" << std::fixed
                      << std::setprecision(4) << loss << " best=" << best
                      << " lr=" << std::setprecision(6) << cur_lr << "\n";
            hima.print_status();
        }

        // Checkpoint — full model save
        if (step % 1000 == 0) {
            std::string ckpt_dir = cfg.getString("CHECKPOINT_DIR", "checkpoints/");
            system(("mkdir -p " + ckpt_dir).c_str());
            save_full_model(ckpt_dir + "model_step_" + std::to_string(step) + ".jlx",
                            emb_lm, tl, dim, layers, heads, vocab, density);
            optimizer.save_state(ckpt_dir + "opt_step_" + std::to_string(step) + ".bin");
        }
    }

    // Final save
    {
        std::string ckpt_dir = cfg.getString("CHECKPOINT_DIR", "checkpoints/");
        system(("mkdir -p " + ckpt_dir).c_str());
        save_full_model(ckpt_dir + "model_final.jlx",
                        emb_lm, tl, dim, layers, heads, vocab, density);
    }

    std::cout << "\n  Training done. best_loss=" << std::fixed
              << std::setprecision(4) << best << " tokens=" << total
              << "\n  Final memory: GPU=" << get_gpu_mb()
              << "MB RSS=" << get_rss_mb() << "MB\n";
    return 0;
}