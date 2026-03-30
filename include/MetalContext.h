#pragma once

#import <Metal/Metal.h>
#import <iostream>

class MetalContext {
public:
    static MetalContext& getInstance() {
        static MetalContext instance;
        return instance;
    }

    id<MTLDevice> getDevice() const { return device; }
    id<MTLCommandQueue> getCommandQueue() const { return commandQueue; }

    MetalContext(const MetalContext&) = delete;
    void operator=(const MetalContext&) = delete;

private:
    MetalContext() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: Failed to create Metal device." << std::endl;
            exit(1);
        }
        commandQueue = [device newCommandQueue];
        std::cout << "Metal Context Initialized: " << [device.name UTF8String] << std::endl;
    }

    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
};
