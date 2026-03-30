#pragma once

#include "Model.h"
#include "Adafactor.h"
#include "Loss.h"
#include "runtime/Quantizer.h"
#include <vector>

class Executor {
public:
    Executor(std::shared_ptr<Model> model, std::shared_ptr<Adafactor> optimizer);

    void train_step(
        std::vector<std::shared_ptr<Tensor>> inputs,
        std::vector<std::shared_ptr<Tensor>> targets,
        std::shared_ptr<Loss> loss_fn);

private:
    std::shared_ptr<Model>     model;
    std::shared_ptr<Adafactor> optimizer;
    std::shared_ptr<Quantizer> quantizer;  // shared, not recreated per step
};
