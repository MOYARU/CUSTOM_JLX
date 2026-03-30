#pragma once

#include <vector>
#include <memory>
#include "Tensor.h"

class Adafactor;  // forward declare

class Layer {
public:
    virtual ~Layer() = default;

    virtual std::vector<std::shared_ptr<Tensor>> forward(
        const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;

    virtual std::vector<std::shared_ptr<Tensor>> backward(
        const std::vector<std::shared_ptr<Tensor>>& grad_outputs) = 0;
        
    virtual void update(Adafactor& optimizer);

    virtual std::vector<std::shared_ptr<Tensor>> parameters()     = 0;
    virtual std::vector<std::shared_ptr<Tensor>> param_gradients() = 0;
    virtual void clear_gradients()   = 0;
    virtual void clear_activations() = 0;

    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is)       = 0;
};
