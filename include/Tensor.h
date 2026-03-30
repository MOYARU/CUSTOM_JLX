#pragma once

#include <vector>
#include <string>
#include <memory>
#import <Metal/Metal.h>

enum class DType {
    Float32,
    BFloat16, // 16-bit brain floating point
    Int8,
    UInt32,   // For indices
    Int32,    // For indices
    Ternary,  // 2-bit packed (-1, 0, 1)
    Bit1      // 1-bit (for extreme quantization if needed)
};

class Tensor {
public:
    Tensor(const std::vector<int>& shape, DType dtype);
    ~Tensor();

    // Delete copy constructor/assignment for now to manage Metal memory simply
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void* data(); // CPU pointer (only if synchronized)
    id<MTLBuffer> getBuffer() const;
    
    size_t size() const; // Number of elements
    size_t bytes() const; // Size in bytes
    
    bool isPacked() const;
    static bool isPacked(DType dtype);
    
    std::vector<int> getShape() const { return shape; }
    DType getDType() const { return dtype; }

    void invalidate(); // Manually release Metal buffer
    void reshape(const std::vector<int>& new_shape);

    static size_t elementSize(DType dtype); 
 // Returns bytes per element (1 for packed types)

private:
    std::vector<int> shape;
    DType dtype;
    id<MTLBuffer> buffer;
    
    void allocate();
};
