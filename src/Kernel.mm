#include "Kernel.h"
#include "MetalContext.h"
#include <fstream>
#include <sstream>
#include <iostream>

// LibraryCache
id<MTLLibrary> LibraryCache::cached_library = nil;
std::string    LibraryCache::cached_path    = "";

id<MTLLibrary> LibraryCache::get(const std::string& source_path) {
    if (cached_library && cached_path == source_path) return cached_library;
    std::ifstream file(source_path);
    if (!file.is_open()) {
        std::cerr << "FATAL: cannot open kernel: " << source_path << "\n"; exit(1);
    }
    std::stringstream buf; buf << file.rdbuf();
    id<MTLDevice> device = MetalContext::getInstance().getDevice();
    NSError* err = nil;
    std::cout << "[Metal] Compiling " << source_path << " ... " << std::flush;
    cached_library = [device newLibraryWithSource:
                      [NSString stringWithUTF8String:buf.str().c_str()]
                      options:[[MTLCompileOptions alloc] init] error:&err];
    if (!cached_library) {
        std::cerr << "\nFATAL: " << [err.localizedDescription UTF8String] << "\n"; exit(1);
    }
    cached_path = source_path;
    std::cout << "ok\n";
    return cached_library;
}

// CommandBatch
CommandBatch& CommandBatch::get() { static CommandBatch b; return b; }

void CommandBatch::begin() {
    if (cmd_) commit_and_wait();
    cmd_ = [MetalContext::getInstance().getCommandQueue() commandBuffer];
    enc_ = nil;
}

id<MTLComputeCommandEncoder> CommandBatch::encoder() {
    if (!enc_) enc_ = [cmd_ computeCommandEncoder];
    return enc_;
}

void CommandBatch::commit_and_wait() {
    if (!cmd_) return;
    if (enc_) { [enc_ endEncoding]; enc_ = nil; }
    [cmd_ commit];
    [cmd_ waitUntilCompleted];
    cmd_ = nil;
}

// Kernel
Kernel::Kernel(const std::string& name, const std::string& source_path)
    : name_(name)
{
    id<MTLLibrary> lib = LibraryCache::get(source_path);
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:
                          [NSString stringWithUTF8String:name_.c_str()]];
    if (!fn) { std::cerr << "FATAL: function not found: " << name_ << "\n"; exit(1); }
    pso_ = [MetalContext::getInstance().getDevice()
            newComputePipelineStateWithFunction:fn error:&err];
    if (!pso_) {
        std::cerr << "FATAL: pipeline " << name_ << ": "
                  << [err.localizedDescription UTF8String] << "\n"; exit(1);
    }
}

// Shared encode logic
static void encode_into(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState>  pso,
                        const std::vector<Tensor*>& outputs,
                        const std::vector<Tensor*>& inputs,
                        int gridX, int gridY,
                        const void* params, size_t param_size)
{
    [enc setComputePipelineState:pso];
    int idx = 0;
    for (auto* t : outputs) [enc setBuffer:t->getBuffer() offset:0 atIndex:idx++];
    for (auto* t : inputs)  [enc setBuffer:t->getBuffer() offset:0 atIndex:idx++];
    if (params) [enc setBytes:params length:param_size atIndex:idx++];
    NSUInteger w  = pso.threadExecutionWidth;
    NSUInteger h  = pso.maxTotalThreadsPerThreadgroup / w;
    NSUInteger gx = (NSUInteger)gridX, gy = (NSUInteger)gridY;
    [enc dispatchThreads:MTLSizeMake(gx, gy, 1)
     threadsPerThreadgroup:MTLSizeMake(MIN(w, gx), MIN(h, gy), 1)];
}

// dispatch: 열린 CommandBatch가 있으면 거기에 encode 아님 없으면 즉시 실행
void Kernel::dispatch(const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs,
                      const void* params, size_t param_size)
{
    if (outputs.empty()) return;
    dispatch2D(inputs, outputs, (int)outputs[0]->size(), 1, params, param_size);
}

void Kernel::dispatch2D(const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs,
                        int gridX, int gridY,
                        const void* params, size_t param_size)
{
    auto& batch = CommandBatch::get();
    if (batch.is_open()) {
        encode_into(batch.encoder(), pso_, outputs, inputs,
                    gridX, gridY, params, param_size);
    } else {
        auto& ctx = MetalContext::getInstance();
        id<MTLCommandBuffer>         cmd = [ctx.getCommandQueue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        encode_into(enc, pso_, outputs, inputs, gridX, gridY, params, param_size);
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    }
}

void Kernel::encode(CommandBatch& batch,
                    const std::vector<Tensor*>& inputs,
                    const std::vector<Tensor*>& outputs,
                    int gridX, int gridY,
                    const void* params, size_t param_size)
{
    encode_into(batch.encoder(), pso_, outputs, inputs,
                gridX, gridY, params, param_size);
}