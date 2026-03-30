#pragma once
#include "Layer.h"
#include "SparseTernaryLinear.h"
#include "Kernel.h"
#include <vector>

class SparseAttention : public Layer {
public:
    SparseAttention(int dim, int num_heads, int block_size, float density);

    std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> backward(
        const std::vector<std::shared_ptr<Tensor>>& grad_outputs) override;

    void update(Adafactor& optimizer) override;
    void resparsify(float density = 0.0f);
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    std::vector<std::shared_ptr<Tensor>> param_gradients() override;
    void clear_gradients() override;
    void clear_activations() override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    std::vector<std::shared_ptr<SparseTernaryLinear>> get_internal_weights() {
        return {w_q, w_k, w_v, w_o};
    }

private:
    int dim, num_heads, head_dim;
    std::shared_ptr<SparseTernaryLinear> w_q, w_k, w_v, w_o;
    std::shared_ptr<Tensor> last_input_;
    std::shared_ptr<Tensor> saved_Q_, saved_K_, saved_V_, saved_attn_;

    // GPU kernels
    std::unique_ptr<Kernel> add_k;
    std::unique_ptr<Kernel> qk_scores_k, softmax_k, attn_v_k;
    std::unique_ptr<Kernel> bwd_dv_k, bwd_dattn_k, bwd_softmax_k, bwd_dq_k, bwd_dk_k;
};