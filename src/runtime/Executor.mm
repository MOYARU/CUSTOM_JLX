#include "runtime/Executor.h"
#include "MetalContext.h"
#include <iostream>

Executor::Executor(std::shared_ptr<Model> model, std::shared_ptr<Adafactor> optimizer)
    : model(model), optimizer(optimizer)
{
    quantizer = std::make_shared<Quantizer>();
}

void Executor::train_step(
    std::vector<std::shared_ptr<Tensor>> inputs,
    std::vector<std::shared_ptr<Tensor>> targets,
    std::shared_ptr<Loss> loss_fn)
{
    // main.mm이 직접 학습 루프를 관리하므로 이 함수는 사용 안함..
    // 호환성을 위해 유지
    (void)inputs;
    (void)targets;
    (void)loss_fn;
}
