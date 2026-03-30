#pragma once

#include "Layer.h"
#include "Kernel.h"
#include <random>

class LinearTernary : public Layer {
public:
    LinearTernary(int in_features, int out_features);
    
    std::vector<std::shared_ptr<Tensor>> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override;
    std::vector<std::shared_ptr<Tensor>> backward(const std::vector<std::shared_ptr<Tensor>>& grad_outputs) override;
    
    std::vector<std::shared_ptr<Tensor>> parameters() override      { return {weights}; }
    std::vector<std::shared_ptr<Tensor>> param_gradients() override { return {weight_grads}; }
    void clear_gradients()  override { weight_grads.reset(); }
    void clear_activations() override {}

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

private:
    int in_features;
    int out_features;
    
    std::shared_ptr<Tensor> weights;      // Ternary (Packed)
    std::shared_ptr<Tensor> weight_grads; // FP32 or BF16 for accumulation
    
    std::unique_ptr<Kernel> matmulKernel;
    
    void initialize_weights();
};
