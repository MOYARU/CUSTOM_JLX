#pragma once

#include "Tensor.h"
#include "Kernel.h"
#include <vector>
#include <memory>

// Abstract base
class Loss {
public:
    virtual ~Loss() = default;
    virtual float forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> backward() = 0;
};

// Metal-backed cross-entropy with fused softmax gradient
class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss();

    // logits: [batch, vocab]  targets: [batch] Int32  logits_grad: [batch, vocab] FP32 (out)
    float forward_full(std::shared_ptr<Tensor> logits,
                       std::shared_ptr<Tensor> targets,
                       std::shared_ptr<Tensor> logits_grad);

    // Loss base interface (stub — use forward_full for real training)
    float forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) override;
    std::vector<std::shared_ptr<Tensor>> backward() override { return {cached_grad}; }

private:
    std::unique_ptr<Kernel> lossKernel;
    std::shared_ptr<Tensor> cached_grad;
};
