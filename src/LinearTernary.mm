#include "LinearTernary.h"
#include <iostream>

LinearTernary::LinearTernary(int in_features, int out_features) 
    : in_features(in_features), out_features(out_features) {
    
    std::vector<int> shape = {out_features, in_features};
    weights = std::make_shared<Tensor>(shape, DType::Ternary);
    
    // Gradients usually need higher precision (Float32 or BF16) before quantization
    weight_grads = std::make_shared<Tensor>(shape, DType::Float32);
    
    initialize_weights();
}

void LinearTernary::initialize_weights() {
    // Fill with random ternary values (-1, 0, 1) packed.
    // Implementation skipped for brevity, filling with 0x00
    // In a real scenario, we'd use a kernel or CPU setup.
}

std::vector<std::shared_ptr<Tensor>> LinearTernary::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.empty()) return {};
    auto input = inputs[0];
    
    // Output shape: [batch, out_features]
    // Input shape: [batch, in_features]
    
    int batch_size = input->getShape()[0];
    std::vector<int> out_shape = {batch_size, out_features};
    auto output = std::make_shared<Tensor>(out_shape, DType::Float32); // Output activations usually float
    
    std::cout << "LinearTernary Forward: " << batch_size << "x" << in_features 
              << " -> " << batch_size << "x" << out_features << std::endl;

    // Dispatch Kernel (Commented out until we have the specific matmul kernel)
    // matmulKernel->dispatch({input.get(), weights.get()}, {output.get()});
    
    return {output};
}

std::vector<std::shared_ptr<Tensor>> LinearTernary::backward(const std::vector<std::shared_ptr<Tensor>>& /*grad_outputs*/) {
    std::cout << "LinearTernary Backward" << std::endl;
    return {}; 
}

void LinearTernary::save(std::ostream& os) const {
    os.write((const char*)&in_features,  sizeof(int));
    os.write((const char*)&out_features, sizeof(int));
    if (weights) os.write((const char*)weights->data(), weights->bytes());
}

void LinearTernary::load(std::istream& is) {
    is.read((char*)&in_features,  sizeof(int));
    is.read((char*)&out_features, sizeof(int));
    if (weights) is.read((char*)weights->data(), weights->bytes());
}