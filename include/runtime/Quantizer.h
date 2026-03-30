#pragma once

#include "Tensor.h"
#include "Kernel.h"
#include <random>

class Quantizer {
public:
    Quantizer();
    
    // TernGrad: Stochastic quantization to Ternary (-1, 0, 1)
    void quantize_terngrad(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output);

private:
    std::unique_ptr<Kernel> terngradKernel;
    std::mt19937 gen;
};
