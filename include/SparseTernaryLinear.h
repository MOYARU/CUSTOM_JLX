#pragma once
#include "Layer.h"
#include "Kernel.h"
#include <vector>

class Adafactor;

class SparseTernaryLinear : public Layer {
public:
    SparseTernaryLinear(int in_features, int out_features, float density);

    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {master_weights_pos, master_weights_neg};
    }
    std::vector<std::shared_ptr<Tensor>> param_gradients() override {
        return {gw_pos, gw_neg};
    }

    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> backward(
        const std::vector<std::shared_ptr<Tensor>>& grad_outputs) override;

    void update(Adafactor& optimizer) override;
    void clear_gradients() override;
    void clear_activations() override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    void sync_packed_weights();
    void fused_adam_update(Adafactor& opt, float grad_scale, float clip_scale, int step = -1);
    void resparsify(float target_density = 0.0f);

    // Accessors
    std::shared_ptr<Tensor> get_master_weights() const { return master_weights_pos; }
    std::shared_ptr<Tensor> get_master_weights_pos() const { return master_weights_pos; }
    std::shared_ptr<Tensor> get_master_weights_neg() const { return master_weights_neg; }
    std::shared_ptr<Tensor> pos_gradients() { return gw_pos; }
    std::shared_ptr<Tensor> neg_gradients() { return gw_neg; }

    uint32_t nnz_pos() const { return _nnz_pos; }
    uint32_t nnz_neg() const { return _nnz_neg; }

    // For optimizer state registration
    std::shared_ptr<Tensor> pos_row_idx, pos_col_idx;
    std::shared_ptr<Tensor> neg_row_idx, neg_col_idx;

private:
    int in_features, out_features;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
    float init_density = 0.01f;
#pragma clang diagnostic pop
    uint32_t _nnz_pos = 0, _nnz_neg = 0;

    // NNZ-indexed master weights (sparse, not dense!) — 1.18GB saved vs full matrix
    std::shared_ptr<Tensor> master_weights_pos;  // [nnz_pos] BF16
    std::shared_ptr<Tensor> master_weights_neg;  // [nnz_neg] BF16

    // Packed weights for GPU forward/backward
    std::shared_ptr<Tensor> packed_pos_w, packed_neg_w;
    std::shared_ptr<Tensor> packed_pos_w_csc, packed_neg_w_csc;

    // CSR (forward)
    std::shared_ptr<Tensor> pos_indices, neg_indices;
    std::shared_ptr<Tensor> row_ptrs, row_counts;

    // CSC (backward input)
    std::shared_ptr<Tensor> pos_indices_col, neg_indices_col;
    std::shared_ptr<Tensor> col_ptrs, col_counts;

    // Prefix-sum pointers
    std::shared_ptr<Tensor> pos_csr_row_ptr, neg_csr_row_ptr;
    std::shared_ptr<Tensor> pos_csc_col_ptr, neg_csc_col_ptr;

    // Gradients
    std::shared_ptr<Tensor> gw_pos, gw_neg;
    std::shared_ptr<Tensor> last_input;

    // Kernels
    std::unique_ptr<Kernel> sparseForwardKernel;
    std::unique_ptr<Kernel> sparseBackwardInputKernel;
    std::unique_ptr<Kernel> sparseBackwardWeightPosKernel;
    std::unique_ptr<Kernel> sparseBackwardWeightNegKernel;
    std::unique_ptr<Kernel> addKernel;

    void initialize_sparse_weights(float density);
};