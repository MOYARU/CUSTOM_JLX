#pragma once
#include "Layer.h"
#include "SparseTernaryLinear.h"
#include "Kernel.h"
#include <vector>

class SparseFFN : public Layer {
public:
    SparseFFN(int dim, float density, int hidden_dim = 0);

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
        return {w_gate, w_up, w_down};
    }

private:
    int dim_, hidden_dim_;
    float density_;

    std::shared_ptr<SparseTernaryLinear> w_gate, w_up, w_down;

    // Cached kernels (created once in constructor)
    std::unique_ptr<Kernel> silu_kernel;
    std::unique_ptr<Kernel> silu_backward_kernel;
    std::unique_ptr<Kernel> mul_kernel;
    std::unique_ptr<Kernel> add3_kernel;

    // Saved for backward
    std::shared_ptr<Tensor> last_input_, gate_out_, up_out_, gate_raw_;
};