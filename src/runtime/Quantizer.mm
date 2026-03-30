#include "Quantizer.h"
#include "MetalContext.h"
#include <cmath>
#include <cstring>

Quantizer::Quantizer() : gen(std::random_device{}()) {
    terngradKernel = std::make_unique<Kernel>("terngrad_quantize_stochastic", "kernels/ops.metal");
}

void Quantizer::quantize_terngrad(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) {
    if (!input || input->size() == 0) return;

    // Compute max |gradient|
    float* data = (float*)input->data();
    float max_abs = 0.f;
    for (size_t i = 0; i < input->size(); ++i) {
        float a = fabsf(data[i]);
        if (a > max_abs) max_abs = a;
    }

    uint32_t n = (uint32_t)input->size();
    uint32_t packed_n = (n + 15) / 16;

    if (!output || output->size() != packed_n)
        output = std::make_shared<Tensor>(std::vector<int>{(int)packed_n}, DType::UInt32);

    uint32_t seed = gen();

    auto enc = CommandBatch::get().encoder();
    [enc setComputePipelineState:terngradKernel->getPipelineState()];
    [enc setBuffer:output->getBuffer() offset:0 atIndex:0];
    [enc setBuffer:input->getBuffer()  offset:0 atIndex:1];
    [enc setBytes:&max_abs length:4 atIndex:2];
    [enc setBytes:&seed    length:4 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(packed_n,1,1)
     threadsPerThreadgroup:MTLSizeMake(std::min(packed_n, (uint32_t)1024),1,1)];
}
