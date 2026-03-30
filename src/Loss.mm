#include "Loss.h"
#include "MetalContext.h"
#include <iostream>

CrossEntropyLoss::CrossEntropyLoss() {
    lossKernel = std::make_unique<Kernel>("cross_entropy_loss", "kernels/ops.metal");
}

float CrossEntropyLoss::forward_full(
    std::shared_ptr<Tensor> logits,
    std::shared_ptr<Tensor> targets,
    std::shared_ptr<Tensor> logits_grad)
{
    int batch_size = (int)targets->size();
    int vocab_size = (int)logits->getShape().back();

    auto loss_tensor = std::make_shared<Tensor>(std::vector<int>{batch_size}, DType::Float32);
    cached_grad = logits_grad;

    id<MTLCommandQueue>          queue = MetalContext::getInstance().getCommandQueue();
    id<MTLCommandBuffer>         cmd   = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc   = [cmd computeCommandEncoder];

    [enc setComputePipelineState:lossKernel->getPipelineState()];
    [enc setBuffer:loss_tensor->getBuffer() offset:0 atIndex:0];  // OUTPUT ls
    [enc setBuffer:logits_grad->getBuffer() offset:0 atIndex:1];  // OUTPUT gs
    [enc setBuffer:logits->getBuffer()      offset:0 atIndex:2];
    [enc setBuffer:targets->getBuffer()     offset:0 atIndex:3];
    uint32_t b = (uint32_t)batch_size, v = (uint32_t)vocab_size;
    [enc setBytes:&b length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&v length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    float* data = (float*)loss_tensor->data();
    float sum = 0.f;
    for (int i = 0; i < batch_size; ++i) sum += data[i];
    return sum / (float)batch_size;
}

// Stub for the abstract interface (used by StubLoss path in main.mm)
float CrossEntropyLoss::forward(
    std::shared_ptr<Tensor> /*pred*/,
    std::shared_ptr<Tensor> /*target*/)
{
    return 0.f;
}