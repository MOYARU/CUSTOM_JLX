#pragma once

#include "Tensor.h"
#include "Kernel.h"
#include <map>
#include <memory>
#include <vector>
#include <unordered_map>

struct AdafactorParams {
    float lr             = 3e-4f;
    float decay_rate     = 0.999f;  // beta2 (second moment)
    float epsilon1       = 1e-30f;
    float epsilon2       = 1e-8f;   // epsilon for v
    float clip_threshold = 1.0f;    // gradient clip
    float beta1          = 0.9f;    // first moment decay
    int   warmup_steps   = 100;
};

class SparseTernaryLinear;

class Adafactor {
public:
    explicit Adafactor(AdafactorParams params = AdafactorParams());
    void set_lr(float lr) { params.lr = lr; }
    float get_lr() const { return params.lr; }

    void save_state(const std::string& path) const;
    void load_state(const std::string& path);

    void register_weight(Tensor* key, int layer_idx) { weight_to_idx[key] = layer_idx; idx_to_weight[layer_idx] = key; }

    void step(std::shared_ptr<Tensor> weights, std::shared_ptr<Tensor> grads);

    void step_sparse(
        std::shared_ptr<Tensor> weights,
        std::shared_ptr<Tensor> gw_pos,
        std::shared_ptr<Tensor> gw_neg,
        std::shared_ptr<Tensor> pos_row_idx,
        std::shared_ptr<Tensor> pos_col_idx,
        std::shared_ptr<Tensor> neg_row_idx,
        std::shared_ptr<Tensor> neg_col_idx,
        std::shared_ptr<Tensor> pos_csr_ptr,
        std::shared_ptr<Tensor> neg_csr_ptr,
        std::shared_ptr<Tensor> pos_csc_ptr,
        std::shared_ptr<Tensor> neg_csc_ptr,
        int out_features, int in_features);

    // Fused GPU: scale + clip + Adam update
    void step_sparse_fused(
        std::shared_ptr<Tensor> weights_pos,
        std::shared_ptr<Tensor> weights_neg,
        std::shared_ptr<Tensor> gw_pos,
        std::shared_ptr<Tensor> gw_neg,
        std::shared_ptr<Tensor> pos_row_idx,
        std::shared_ptr<Tensor> pos_col_idx,
        std::shared_ptr<Tensor> neg_row_idx,
        std::shared_ptr<Tensor> neg_col_idx,
        int in_features,
        float grad_scale,
        float clip_scale,
        int step_count_override = -1);

private:
    AdafactorParams params;

    // Per-NNZ state (sparse layers)
    struct SparseState1bit {
        std::vector<float> m_pos;     // [nnz_pos] F32 first moment (CPU, for save/load)
        std::vector<float> m_neg;
        std::vector<float> v_pos;     // [nnz_pos] F32 second moment
        std::vector<float> v_neg;
        std::vector<uint32_t> m1_pos; // 1-bit packed (phase 2)
        std::vector<uint32_t> m1_neg;
        float m_scale_pos = 0.f;
        float m_scale_neg = 0.f;
        int step_count = 0;
        bool phase2 = false;
        // GPU tensors for fused kernel (lazily allocated)
        std::shared_ptr<Tensor> m_pos_gpu, m_neg_gpu;
        std::shared_ptr<Tensor> v_pos_gpu, v_neg_gpu;
    };

    // Dense state (unchanged)
    struct DenseState {
        std::shared_ptr<Tensor> row_v;
        std::shared_ptr<Tensor> col_v;
        int step_count = 0;
    };

    std::unordered_map<Tensor*, SparseState1bit> sparse1bit_states;
    std::map<Tensor*, DenseState> dense_states;
    std::unordered_map<Tensor*, int> weight_to_idx;
    std::unordered_map<int, Tensor*> idx_to_weight;

    // Dense kernels (kept for non-sparse layers)
    std::unique_ptr<Kernel> k_reduce_row_bf16;
    std::unique_ptr<Kernel> k_reduce_col_bf16;
    std::unique_ptr<Kernel> k_update_bf16;
    std::unique_ptr<Kernel> k_fill_bf16;
    std::unique_ptr<Kernel> k_reduce_row;
    std::unique_ptr<Kernel> k_reduce_col;
    std::unique_ptr<Kernel> k_update;

    // Fused GPU kernel
    std::unique_ptr<Kernel> k_fused_adam_sparse;
};