#include "Tensor.h"
#include "MetalContext.h"
#include <numeric>
#include <cmath>
#include <cstring>
#include <iostream>

Tensor::Tensor(const std::vector<int>& shape, DType dtype) 
    : shape(shape), dtype(dtype) {
    allocate();
}

Tensor::~Tensor() {
    if (buffer) {
        buffer = nil;
    }
}

size_t Tensor::elementSize(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::BFloat16: return 2;
        case DType::Int8: return 1;
        case DType::UInt32: return 4;
        case DType::Int32: return 4;
        case DType::Ternary: return 1; // 4 elements per byte
        case DType::Bit1: return 1;    // 8 elements per byte
    }
    return 0;
}

bool Tensor::isPacked() const {
    return isPacked(dtype);
}

bool Tensor::isPacked(DType dtype) {
    return dtype == DType::Ternary || dtype == DType::Bit1;
}

size_t Tensor::size() const {
    if (shape.empty()) return 0;
    size_t s = 1;
    for (int d : shape) s *= (size_t)d;
    return s;
}

size_t Tensor::bytes() const {
    size_t num_elements = size();
    switch (dtype) {
        case DType::Float32: return num_elements * 4ULL;
        case DType::BFloat16: return num_elements * 2ULL;
        case DType::Int8: return num_elements * 1ULL;
        case DType::UInt32: return num_elements * 4ULL;
        case DType::Int32: return num_elements * 4ULL;
        case DType::Ternary: 
            return (num_elements + 3ULL) / 4ULL; 
        case DType::Bit1:
            return (num_elements + 7ULL) / 8ULL;
    }
    return 0;
}

void Tensor::allocate() {
    size_t byte_size = bytes();
    if (byte_size == 0) return;

    if (byte_size > 8ULL * 1024 * 1024 * 1024) {
        std::cerr << "FATAL: Allocation too large (" << (byte_size/(1024*1024)) << " MB)"
                  << "  dtype=" << (int)dtype
                  << "  shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cerr << shape[i];
            if (i + 1 < shape.size()) std::cerr << ",";
        }
        std::cerr << "]  size()=" << size() << std::endl;
        exit(1);
    }

    id<MTLDevice> device = MetalContext::getInstance().getDevice();
    buffer = [device newBufferWithLength:byte_size options:MTLResourceStorageModeShared];
    
    if (!buffer) {
        std::cerr << "Failed to allocate Metal buffer of size " << byte_size << std::endl;
        exit(1);
    }
}

id<MTLBuffer> Tensor::getBuffer() const {
    return buffer;
}

void Tensor::invalidate() {
    if (buffer) {
        buffer = nil;
    }
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    // Check if total size remains the same
    size_t new_size = 1;
    for (int d : new_shape) new_size *= (size_t)d;
    if (new_size == size()) {
        shape = new_shape;
    }
}

void* Tensor::data() {
    return [buffer contents];
}