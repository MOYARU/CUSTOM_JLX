#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

#include "MetalContext.h"
#include "Kernel.h"
#include "Tensor.h"
#include "SparseAttention.h"
#include "SparseFFN.h"
#include "SparseTernaryLinear.h"

// MPS matmul helper
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

// RMSNorm
struct RMSNorm {
    float eps = 1e-6f;
    std::unique_ptr<Kernel> fwd_k;
    void init() { fwd_k = std::make_unique<Kernel>("rms_norm_forward", "kernels/ops.metal"); }
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
};

// Factored Emb+LM Head (inference only)
struct FactoredEmbLMHead {
    int vocab, dim, k;
    std::vector<float> sub_emb;          // [vocab × k] CPU
    std::shared_ptr<Tensor> sub_emb_gpu; // [vocab × k] GPU
    std::shared_ptr<Tensor> emb_proj_gpu;// [k × dim] GPU
    std::shared_ptr<Tensor> lm_proj_gpu; // [dim × k] GPU

    void init(int vocab_, int dim_, int k_) {
        vocab = vocab_; dim = dim_; k = k_;
        sub_emb.resize((size_t)vocab * k);
        sub_emb_gpu = std::make_shared<Tensor>(std::vector<int>{vocab, k}, DType::Float32);
        emb_proj_gpu = std::make_shared<Tensor>(std::vector<int>{k, dim}, DType::Float32);
        lm_proj_gpu = std::make_shared<Tensor>(std::vector<int>{dim, k}, DType::Float32);
    }

    void load(std::istream& f) {
        int k_file;
        f.read((char*)&k_file, sizeof(int));
        if (k_file != k) { std::cerr << "k mismatch\n"; return; }
        f.read((char*)sub_emb.data(), sub_emb.size() * 4);
        std::vector<float> emb_proj((size_t)k * dim);
        f.read((char*)emb_proj.data(), emb_proj.size() * 4);
        std::vector<float> lm_proj((size_t)dim * k);
        f.read((char*)lm_proj.data(), lm_proj.size() * 4);
        std::memcpy(sub_emb_gpu->data(), sub_emb.data(), sub_emb.size() * 4);
        std::memcpy(emb_proj_gpu->data(), emb_proj.data(), emb_proj.size() * 4);
        std::memcpy(lm_proj_gpu->data(), lm_proj.data(), lm_proj.size() * 4);
    }

    // Embedding: token → sub_emb lookup → emb_proj matmul → + pos_enc
    std::shared_ptr<Tensor> emb_forward(
        const std::vector<int32_t>& tokens, const std::vector<float>& pos_enc,
        int cur_len, int seq)
    {
        auto sub_out = std::make_shared<Tensor>(std::vector<int>{cur_len, k}, DType::Float32);
        float* sp = (float*)sub_out->data();
        for (int i = 0; i < cur_len; ++i) {
            int v = std::max(0, std::min((int)tokens[i], vocab - 1));
            std::memcpy(sp + i * k, sub_emb.data() + (size_t)v * k, k * sizeof(float));
        }
        auto out = std::make_shared<Tensor>(std::vector<int>{cur_len, dim}, DType::Float32);
        mps_matmul(sub_out->getBuffer(), emb_proj_gpu->getBuffer(), out->getBuffer(),
                   cur_len, dim, k);
        float* op = (float*)out->data();
        for (int i = 0; i < cur_len; ++i) {
            const float* pe = pos_enc.data() + (size_t)(i % seq) * dim;
            for (int d = 0; d < dim; ++d) op[i * dim + d] += pe[d];
        }
        return out;
    }

    // LM head: hidden → lm_proj → sub_emb^T → logits
    std::shared_ptr<Tensor> lm_forward(std::shared_ptr<Tensor> hidden, int BS) {
        auto projected = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
        mps_matmul(hidden->getBuffer(), lm_proj_gpu->getBuffer(), projected->getBuffer(),
                   BS, k, dim);
        auto logits = std::make_shared<Tensor>(std::vector<int>{BS, vocab}, DType::Float32);
        mps_matmul(projected->getBuffer(), sub_emb_gpu->getBuffer(), logits->getBuffer(),
                   BS, vocab, k, false, true);
        return logits;
    }
};

// Sparse Transformer Layer
struct SparseTransformerLayer {
    RMSNorm norm1, norm2;
    std::shared_ptr<SparseAttention> attn;
    std::shared_ptr<SparseFFN> ffn;
    void init(int dim, int num_heads, int block_size, float density, int ffn_dim) {
        norm1.init(); norm2.init();
        attn = std::make_shared<SparseAttention>(dim, num_heads, block_size, density);
        ffn  = std::make_shared<SparseFFN>(dim, density, ffn_dim);
    }
};

// GPU add
static void gpu_add(Kernel& k, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    uint32_t n = (uint32_t)a->size();
    auto enc = CommandBatch::get().encoder();
    [enc setComputePipelineState:k.getPipelineState()];
    [enc setBuffer:a->getBuffer() offset:0 atIndex:0];
    [enc setBuffer:b->getBuffer() offset:0 atIndex:1];
    [enc setBytes:&n length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

// Positional encoding
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

// op-k sampling
static int sample_top_k(const float* logits, int vocab, float temp, int top_k,
                         std::mt19937& rng) {
    std::vector<std::pair<float, int>> vals(vocab);
    for (int i = 0; i < vocab; ++i) vals[i] = {logits[i], i};
    std::partial_sort(vals.begin(), vals.begin() + top_k, vals.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    float mx = vals[0].first, sum = 0.f;
    std::vector<float> probs(top_k);
    for (int i = 0; i < top_k; ++i) {
        probs[i] = expf((vals[i].first - mx) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; ++i) probs[i] /= sum;
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(rng), c = 0.f;
    for (int i = 0; i < top_k; ++i) { c += probs[i]; if (r <= c) return vals[i].second; }
    return vals[0].second;
}

// ㅡㅡ
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " model.jlx [max_tokens=128] [temperature=0.8] [top_k=40]\n";
        return 1;
    }

    std::string model_path = argv[1];
    int max_gen = (argc > 2) ? atoi(argv[2]) : 128;
    float temperature = (argc > 3) ? atof(argv[3]) : 0.8f;
    int top_k = (argc > 4) ? atoi(argv[4]) : 40;
    int seq = 256;

    // Load model header
    std::ifstream f(model_path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << model_path << "\n"; return 1; }

    uint32_t magic;
    f.read((char*)&magic, 4);
    if (magic != 0x4A4C5832) { std::cerr << "Bad magic (need JLX2)\n"; return 1; }

    int dim, layers, heads, vocab;
    float density;
    f.read((char*)&dim, sizeof(int));
    f.read((char*)&layers, sizeof(int));
    f.read((char*)&heads, sizeof(int));
    f.read((char*)&vocab, sizeof(int));
    f.read((char*)&density, sizeof(float));

    std::cerr << "Model: dim=" << dim << " layers=" << layers
              << " heads=" << heads << " vocab=" << vocab
              << " density=" << density << "\n";

    int ffn_dim = dim * 4;

    // Factored Emb+LM head
    // Read k from the save stream
    int emb_k;
    auto pos = f.tellg();
    f.read((char*)&emb_k, sizeof(int));
    f.seekg(pos);  // rewind — load() reads k again

    FactoredEmbLMHead emb_lm;
    emb_lm.init(vocab, dim, emb_k);
    emb_lm.load(f);
    std::cerr << "Factored emb: k=" << emb_k << "\n";

    // Sparse layers
    std::vector<SparseTransformerLayer> tl(layers);
    for (int i = 0; i < layers; ++i)
        tl[i].init(dim, heads, 32, density, ffn_dim);
    for (int i = 0; i < layers; ++i) {
        tl[i].attn->load(f);
        tl[i].ffn->load(f);
    }
    for (int i = 0; i < layers; ++i) {
        for (auto& w : tl[i].attn->get_internal_weights()) w->sync_packed_weights();
        for (auto& w : tl[i].ffn->get_internal_weights()) w->sync_packed_weights();
    }
    f.close();

    RMSNorm final_norm; final_norm.init();
    Kernel add_k("elementwise_add", "kernels/ops.metal");
    auto pos_enc = make_sinusoidal(seq, dim);

    // Read prompt tokens
    std::vector<int32_t> tokens;
    { std::string line; std::getline(std::cin, line);
      std::istringstream iss(line); int tok;
      while (iss >> tok) tokens.push_back(tok); }
    if (tokens.empty()) { std::cerr << "No input tokens\n"; return 1; }

    std::cerr << "Prompt: " << tokens.size() << " tokens, generating " << max_gen << "\n";
    std::mt19937 rng(42);

    // Autoregressive generation
    for (int g = 0; g < max_gen; ++g) {
        int cur_len = std::min((int)tokens.size(), seq);
        int start = (int)tokens.size() - cur_len;

        // Slice tokens for current window
        std::vector<int32_t> window(tokens.begin() + start, tokens.begin() + start + cur_len);

        // Factored embedding forward
        auto r = emb_lm.emb_forward(window, pos_enc, cur_len, seq);

        // Forward through layers
        for (int li = 0; li < layers; ++li) {
            auto& L = tl[li];
            r->reshape({1, cur_len, dim});
            CommandBatch::get().begin();
            auto xn = L.norm1.forward(r);
            xn->reshape({1, cur_len, dim});
            auto attn_out = L.attn->forward({xn})[0];
            attn_out->reshape({cur_len, dim});
            r->reshape({cur_len, dim});
            gpu_add(add_k, r, attn_out);
            CommandBatch::get().commit_and_wait();

            r->reshape({1, cur_len, dim});
            CommandBatch::get().begin();
            auto xn2 = L.norm2.forward(r);
            xn2->reshape({1, cur_len, dim});
            auto ffn_out = L.ffn->forward({xn2})[0];
            ffn_out->reshape({cur_len, dim});
            r->reshape({cur_len, dim});
            gpu_add(add_k, r, ffn_out);
            CommandBatch::get().commit_and_wait();

            r->reshape({1, cur_len, dim});
            L.attn->clear_activations();
            L.ffn->clear_activations();
        }

        // Final norm
        CommandBatch::get().begin();
        r->reshape({cur_len, dim});
        auto xf = final_norm.forward(r);
        CommandBatch::get().commit_and_wait();

        // Extract last position
        auto last_h = std::make_shared<Tensor>(std::vector<int>{1, dim}, DType::Float32);
        std::memcpy(last_h->data(), (float*)xf->data() + (cur_len - 1) * dim,
                     dim * sizeof(float));

        // Factored LM head
        auto logits = emb_lm.lm_forward(last_h, 1);

        int next = sample_top_k((float*)logits->data(), vocab, temperature, top_k, rng);
        tokens.push_back(next);
        std::cout << next << "\n";
        std::cout.flush();
    }

    return 0;
}