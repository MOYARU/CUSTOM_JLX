#pragma once
#include <string>
#include <vector>
#import <Metal/Metal.h>
#include "Tensor.h"

// ── Shared MTLLibrary — compiled once per process ─────────────────────────
class LibraryCache {
public:
    static id<MTLLibrary> get(const std::string& source_path);
private:
    static id<MTLLibrary> cached_library;
    static std::string    cached_path;
};
// Usage:
//   auto& batch = CommandBatch::get();
//   batch.begin();                     // once before a layer
//   kernel.encode(batch, ...);         // many kernels, no sync
//   batch.commit_and_wait();           // one GPU sync at end of layer
class CommandBatch {
public:
    static CommandBatch& get();        // process-wide singleton

    void begin();                      // new commandBuffer
    id<MTLComputeCommandEncoder> encoder(); // lazy encoder
    void commit_and_wait();            // endEncoding + commit + wait

    bool is_open() const { return cmd_ != nil; }

private:
    CommandBatch() = default;
    id<MTLCommandBuffer>         cmd_ = nil;
    id<MTLComputeCommandEncoder> enc_ = nil;
};

// ─────────────────────────────────────────────────────────────────────────
class Kernel {
public:
    Kernel(const std::string& name, const std::string& source_path);

    // Immediate dispatch: own commandBuffer + waitUntilCompleted
    void dispatch(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  const void* params = nullptr, size_t param_size = 0);

    void dispatch2D(const std::vector<Tensor*>& inputs,
                    const std::vector<Tensor*>& outputs,
                    int gridX, int gridY,
                    const void* params = nullptr, size_t param_size = 0);

    // Batched encode: write into the open CommandBatch encoder, no sync
    void encode(CommandBatch& batch,
                const std::vector<Tensor*>& inputs,
                const std::vector<Tensor*>& outputs,
                int gridX, int gridY,
                const void* params = nullptr, size_t param_size = 0);

    id<MTLComputePipelineState> getPipelineState() const { return pso_; }

private:
    std::string name_;
    id<MTLComputePipelineState> pso_;
    void compile(const std::string& source_path);
};
