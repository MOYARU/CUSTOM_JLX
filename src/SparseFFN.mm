#include "SparseFFN.h"
#include "Adafactor.h"
#include "MetalContext.h"
#include <iostream>
#include <cmath>
#include <cstring>

SparseFFN::SparseFFN(int dim, float density, int hidden_dim)
    : dim_(dim), density_(density)
{
    if (hidden_dim <= 0) {
        hidden_dim_ = ((dim * 8 / 3) + 63) / 64 * 64;
    } else {
        hidden_dim_ = hidden_dim;
    }

    std::cout << "  SparseFFN [" << dim_ << " → " << hidden_dim_
              << " → " << dim_ << "] density=" << density_ << "\n";

    w_gate = std::make_shared<SparseTernaryLinear>(dim_, hidden_dim_, density_);
    w_up   = std::make_shared<SparseTernaryLinear>(dim_, hidden_dim_, density_);
    w_down = std::make_shared<SparseTernaryLinear>(hidden_dim_, dim_, density_);

    const std::string kpath = "kernels/ops.metal";
    silu_kernel          = std::make_unique<Kernel>("silu_forward",  kpath);
    silu_backward_kernel = std::make_unique<Kernel>("silu_backward", kpath);
    mul_kernel           = std::make_unique<Kernel>("elementwise_mul", kpath);
    add3_kernel          = std::make_unique<Kernel>("elementwise_add3", kpath);
}

// ── Forward: SwiGLU ─────────────────────────────────────────────────────────
std::vector<std::shared_ptr<Tensor>> SparseFFN::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs)
{
    if (inputs.empty()) return {};
    auto x = inputs[0];
    last_input_ = x;

    // W_gate @ x, W_up @ x
    gate_raw_ = w_gate->forward({x})[0];
    up_out_   = w_up->forward({x})[0];

    // SiLU(gate_raw)
    gate_out_ = std::make_shared<Tensor>(gate_raw_->getShape(), DType::Float32);
    {
        uint32_t n = (uint32_t)gate_raw_->size();
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:silu_kernel->getPipelineState()];
        [enc setBuffer:gate_out_->getBuffer()  offset:0 atIndex:0];
        [enc setBuffer:gate_raw_->getBuffer()  offset:0 atIndex:1];
        [enc setBytes:&n length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n,1,1)
         threadsPerThreadgroup:MTLSizeMake(std::min(n,(uint32_t)1024),1,1)];
    }

    // h = gate_out * up_out
    auto h = std::make_shared<Tensor>(gate_out_->getShape(), DType::Float32);
    {
        uint32_t n = (uint32_t)h->size();
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:mul_kernel->getPipelineState()];
        [enc setBuffer:h->getBuffer()          offset:0 atIndex:0];
        [enc setBuffer:gate_out_->getBuffer()  offset:0 atIndex:1];
        [enc setBuffer:up_out_->getBuffer()    offset:0 atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n,1,1)
         threadsPerThreadgroup:MTLSizeMake(std::min(n,(uint32_t)1024),1,1)];
    }

    return w_down->forward({h});
}

// ── Backward ────────────────────────────────────────────────────────────────
std::vector<std::shared_ptr<Tensor>> SparseFFN::backward(
    const std::vector<std::shared_ptr<Tensor>>& grad_outputs)
{
    if (grad_outputs.empty()) return {};
    auto dy = grad_outputs[0];

    auto dh = w_down->backward({dy})[0];

    // d_gate_out = dh * up_out
    auto d_gate_out = std::make_shared<Tensor>(dh->getShape(), DType::Float32);
    {
        uint32_t n = (uint32_t)dh->size();
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:mul_kernel->getPipelineState()];
        [enc setBuffer:d_gate_out->getBuffer() offset:0 atIndex:0];
        [enc setBuffer:dh->getBuffer()         offset:0 atIndex:1];
        [enc setBuffer:up_out_->getBuffer()    offset:0 atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n,1,1)
         threadsPerThreadgroup:MTLSizeMake(std::min(n,(uint32_t)1024),1,1)];
    }

    // d_up_out = dh * gate_out
    auto d_up_out = std::make_shared<Tensor>(dh->getShape(), DType::Float32);
    {
        uint32_t n = (uint32_t)dh->size();
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:mul_kernel->getPipelineState()];
        [enc setBuffer:d_up_out->getBuffer() offset:0 atIndex:0];
        [enc setBuffer:dh->getBuffer()       offset:0 atIndex:1];
        [enc setBuffer:gate_out_->getBuffer()offset:0 atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n,1,1)
         threadsPerThreadgroup:MTLSizeMake(std::min(n,(uint32_t)1024),1,1)];
    }

    // d_gate_raw = d_gate_out * SiLU'(gate_raw)
    auto d_gate_raw = std::make_shared<Tensor>(d_gate_out->getShape(), DType::Float32);
    {
        uint32_t n = (uint32_t)d_gate_out->size();
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:silu_backward_kernel->getPipelineState()];
        [enc setBuffer:d_gate_raw->getBuffer()  offset:0 atIndex:0];
        [enc setBuffer:d_gate_out->getBuffer()  offset:0 atIndex:1];
        [enc setBuffer:gate_raw_->getBuffer()   offset:0 atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n,1,1)
         threadsPerThreadgroup:MTLSizeMake(std::min(n,(uint32_t)1024),1,1)];
    }

    auto dx_gate = w_gate->backward({d_gate_raw})[0];
    auto dx_up   = w_up->backward({d_up_out})[0];

    // dx = dx_gate + dx_up
    auto dx = std::make_shared<Tensor>(dx_gate->getShape(), DType::Float32);
    {
        uint32_t n = (uint32_t)dx->size();
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:add3_kernel->getPipelineState()];
        [enc setBuffer:dx->getBuffer()      offset:0 atIndex:0];
        [enc setBuffer:dx_gate->getBuffer() offset:0 atIndex:1];
        [enc setBuffer:dx_up->getBuffer()   offset:0 atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n,1,1)
         threadsPerThreadgroup:MTLSizeMake(std::min(n,(uint32_t)1024),1,1)];
    }

    return {dx};
}

void SparseFFN::update(Adafactor& /*optimizer*/) {}

void SparseFFN::resparsify(float density) {
    w_gate->resparsify(density);
    w_up->resparsify(density);
    w_down->resparsify(density);
}

std::vector<std::shared_ptr<Tensor>> SparseFFN::parameters() {
    auto pg = w_gate->parameters();
    auto pu = w_up->parameters();
    auto pd = w_down->parameters();
    pg.insert(pg.end(), pu.begin(), pu.end());
    pg.insert(pg.end(), pd.begin(), pd.end());
    return pg;
}

std::vector<std::shared_ptr<Tensor>> SparseFFN::param_gradients() {
    auto gg = w_gate->param_gradients();
    auto gu = w_up->param_gradients();
    auto gd = w_down->param_gradients();
    gg.insert(gg.end(), gu.begin(), gu.end());
    gg.insert(gg.end(), gd.begin(), gd.end());
    return gg;
}

void SparseFFN::clear_gradients() {
    w_gate->clear_gradients(); w_up->clear_gradients(); w_down->clear_gradients();
}

void SparseFFN::clear_activations() {
    last_input_.reset(); gate_out_.reset(); up_out_.reset(); gate_raw_.reset();
    w_gate->clear_activations(); w_up->clear_activations(); w_down->clear_activations();
}

void SparseFFN::save(std::ostream& os) const {
    os.write((const char*)&dim_, sizeof(int));
    os.write((const char*)&hidden_dim_, sizeof(int));
    w_gate->save(os); w_up->save(os); w_down->save(os);
}

void SparseFFN::load(std::istream& is) {
    is.read((char*)&dim_, sizeof(int));
    is.read((char*)&hidden_dim_, sizeof(int));
    w_gate->load(is); w_up->load(is); w_down->load(is);
}