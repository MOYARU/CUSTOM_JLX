#include "Model.h"
#include "SparseTernaryLinear.h"
#include <fstream>
#include <iostream>

void Model::save(const std::string& filepath) const {
    std::ofstream f(filepath, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << filepath << "\n"; return; }
    for (auto& layer : layers) layer->save(f);
    std::cout << "Saved checkpoint: " << filepath << "\n";
}

void Model::load(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << filepath << "\n"; return; }
    for (auto& layer : layers) layer->load(f);
    std::cout << "Loaded checkpoint: " << filepath << "\n";
}

// save , load model