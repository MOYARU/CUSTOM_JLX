#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include "Layer.h"
#include "Adafactor.h"

class Model {
public:
    Model() = default;

    void add_layer(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    std::vector<std::shared_ptr<Tensor>> forward(
        std::vector<std::shared_ptr<Tensor>> inputs)
    {
        for (auto& layer : layers)
            inputs = layer->forward(inputs);
        return inputs;
    }

    std::vector<std::shared_ptr<Layer>>& get_layers() { return layers; }
    const std::vector<std::shared_ptr<Layer>>& get_layers() const { return layers; }

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

private:
    std::vector<std::shared_ptr<Layer>> layers;
};