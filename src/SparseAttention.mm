#include "SparseAttention.h"
#include "MetalContext.h"
#include <iostream>
#include <cmath>
#include <cstring>

SparseAttention::SparseAttention(int dim, int num_heads, int /*block_size*/, float density)
    : dim(dim), num_heads(num_heads)
{
    head_dim = dim / num_heads;
    w_q = std::make_shared<SparseTernaryLinear>(dim, dim, density);
    w_k = std::make_shared<SparseTernaryLinear>(dim, dim, density);
    w_v = std::make_shared<SparseTernaryLinear>(dim, dim, density);
    w_o = std::make_shared<SparseTernaryLinear>(dim, dim, density);

    const std::string kp = "kernels/ops.metal";
    add_k       = std::make_unique<Kernel>("elementwise_add",    kp);
    qk_scores_k = std::make_unique<Kernel>("causal_qk_scores",  kp);
    softmax_k   = std::make_unique<Kernel>("row_softmax",        kp);
    attn_v_k    = std::make_unique<Kernel>("attn_weighted_sum",  kp);
    bwd_dv_k     = std::make_unique<Kernel>("attn_bwd_dv",        kp);
    bwd_dattn_k  = std::make_unique<Kernel>("attn_bwd_dattn",    kp);
    bwd_softmax_k= std::make_unique<Kernel>("attn_bwd_softmax",  kp);
    bwd_dq_k     = std::make_unique<Kernel>("attn_bwd_dq_matmul",kp);
    bwd_dk_k     = std::make_unique<Kernel>("attn_bwd_dk_matmul",kp);
}

std::vector<std::shared_ptr<Tensor>> SparseAttention::forward(
    const std::vector<std::shared_ptr<Tensor>>& inputs)
{
    auto input = inputs[0];
    last_input_ = input;
    auto in_shape = input->getShape();
    int batch = in_shape[0], seq_len = in_shape[1];
    int bh = batch * num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Sparse projections
    auto Q = w_q->forward({input})[0];
    auto K = w_k->forward({input})[0];
    auto V = w_v->forward({input})[0];

    Q->reshape({bh, seq_len, head_dim});
    K->reshape({bh, seq_len, head_dim});
    V->reshape({bh, seq_len, head_dim});

    saved_Q_ = Q; saved_K_ = K; saved_V_ = V;

    // GPU: QK^T scores
    saved_attn_ = std::make_shared<Tensor>(
        std::vector<int>{bh, seq_len, seq_len}, DType::Float32);
    struct P { uint32_t bh, S, hd; float scale; uint32_t window; } p = {
        (uint32_t)bh, (uint32_t)seq_len, (uint32_t)head_dim, scale, 0u };

    { auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:qk_scores_k->getPipelineState()];
      [enc setBuffer:saved_attn_->getBuffer() offset:0 atIndex:0];
      [enc setBuffer:Q->getBuffer()           offset:0 atIndex:1];
      [enc setBuffer:K->getBuffer()           offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(seq_len, seq_len, bh)
       threadsPerThreadgroup:MTLSizeMake(std::min(seq_len,16), std::min(seq_len,16), 1)];
    }

    // GPU: row softmax
    { auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:softmax_k->getPipelineState()];
      [enc setBuffer:saved_attn_->getBuffer() offset:0 atIndex:0];
      [enc setBytes:&p length:sizeof(p) atIndex:1];
      [enc dispatchThreads:MTLSizeMake(seq_len, bh, 1)
       threadsPerThreadgroup:MTLSizeMake(std::min(seq_len,256), 1, 1)];
    }

    // GPU: attn @ V
    auto context = std::make_shared<Tensor>(
        std::vector<int>{bh, seq_len, head_dim}, DType::Float32);
    { auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:attn_v_k->getPipelineState()];
      [enc setBuffer:context->getBuffer()     offset:0 atIndex:0];
      [enc setBuffer:saved_attn_->getBuffer() offset:0 atIndex:1];
      [enc setBuffer:V->getBuffer()           offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(head_dim, seq_len, bh)
       threadsPerThreadgroup:MTLSizeMake(std::min(head_dim,64), 1, 1)];
    }

    // Output projection
    context->reshape({batch, seq_len, dim});
    auto output = w_o->forward({context})[0];
    context->invalidate();
    return {output};
}

std::vector<std::shared_ptr<Tensor>> SparseAttention::backward(
    const std::vector<std::shared_ptr<Tensor>>& grad_outputs)
{
    if (grad_outputs.empty()) return {};
    auto grad_out = grad_outputs[0];
    auto go_shape = grad_out->getShape();
    int batch = go_shape[0], seq_len = go_shape[1];
    int bh = batch * num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    auto grad_context = w_o->backward({grad_out})[0];
    grad_context->reshape({bh, seq_len, head_dim});

    struct P { uint32_t bh, S, hd; float scale; uint32_t window; } p = {
        (uint32_t)bh, (uint32_t)seq_len, (uint32_t)head_dim, scale, 0u };

    // dV
    auto dV = std::make_shared<Tensor>(std::vector<int>{bh, seq_len, head_dim}, DType::Float32);
    { auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:bwd_dv_k->getPipelineState()];
      [enc setBuffer:dV->getBuffer()              offset:0 atIndex:0];
      [enc setBuffer:saved_attn_->getBuffer()     offset:0 atIndex:1];
      [enc setBuffer:grad_context->getBuffer()    offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(head_dim, seq_len, bh)
       threadsPerThreadgroup:MTLSizeMake(std::min(head_dim,64), 1, 1)];
    }

    // 1: d_attn[h,qi,ki] = dOut[h,qi,:] · V[h,ki,:] 
    auto da = std::make_shared<Tensor>(std::vector<int>{bh, seq_len, seq_len}, DType::Float32);
    { auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:bwd_dattn_k->getPipelineState()];
      [enc setBuffer:da->getBuffer()             offset:0 atIndex:0];
      [enc setBuffer:grad_context->getBuffer()   offset:0 atIndex:1];
      [enc setBuffer:saved_V_->getBuffer()       offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(seq_len, seq_len, bh)
       threadsPerThreadgroup:MTLSizeMake(std::min(seq_len,16), std::min(seq_len,16), 1)];
    }

    // 2: softmax backward: ds = attn * (da - rowdot) * scale (in-place on da)
    { auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:bwd_softmax_k->getPipelineState()];
      [enc setBuffer:da->getBuffer()             offset:0 atIndex:0];
      [enc setBuffer:saved_attn_->getBuffer()    offset:0 atIndex:1];
      [enc setBytes:&p length:sizeof(p) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(seq_len, bh, 1)
       threadsPerThreadgroup:MTLSizeMake(std::min(seq_len,256), 1, 1)];
    }

    // 3: dQ[h,qi,d] = sum_ki ds[h,qi,ki] * K[h,ki,d]
    auto dQ = std::make_shared<Tensor>(std::vector<int>{bh, seq_len, head_dim}, DType::Float32);
    { auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:bwd_dq_k->getPipelineState()];
      [enc setBuffer:dQ->getBuffer()          offset:0 atIndex:0];
      [enc setBuffer:da->getBuffer()          offset:0 atIndex:1];
      [enc setBuffer:saved_K_->getBuffer()    offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(head_dim, seq_len, bh)
       threadsPerThreadgroup:MTLSizeMake(std::min(head_dim,64), 1, 1)];
    }

    // 4: dK[h,ki,d] = sum_qi ds[h,qi,ki] * Q[h,qi,d]
    auto dK = std::make_shared<Tensor>(std::vector<int>{bh, seq_len, head_dim}, DType::Float32);
    { auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:bwd_dk_k->getPipelineState()];
      [enc setBuffer:dK->getBuffer()          offset:0 atIndex:0];
      [enc setBuffer:da->getBuffer()          offset:0 atIndex:1];
      [enc setBuffer:saved_Q_->getBuffer()    offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(head_dim, seq_len, bh)
       threadsPerThreadgroup:MTLSizeMake(std::min(head_dim,64), 1, 1)];
    }
    da->invalidate();

    CommandBatch::get().commit_and_wait();
    grad_context->invalidate();
    saved_Q_->invalidate(); saved_K_->invalidate();
    saved_V_->invalidate(); saved_attn_->invalidate();
    saved_Q_.reset(); saved_K_.reset(); saved_V_.reset(); saved_attn_.reset();

    dQ->reshape({batch, seq_len, dim});
    dK->reshape({batch, seq_len, dim});
    dV->reshape({batch, seq_len, dim});

    CommandBatch::get().begin();
    auto gi_q = w_q->backward({dQ})[0];
    auto gi_k = w_k->backward({dK})[0];
    auto gi_v = w_v->backward({dV})[0];

    gi_q->reshape({batch, seq_len, dim});
    gi_k->reshape({batch, seq_len, dim});
    gi_v->reshape({batch, seq_len, dim});

    { uint32_t n = (uint32_t)gi_q->size();
      auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:add_k->getPipelineState()];
      [enc setBuffer:gi_q->getBuffer() offset:0 atIndex:0];
      [enc setBuffer:gi_k->getBuffer() offset:0 atIndex:1];
      [enc setBytes:&n length:4 atIndex:2];
      [enc dispatchThreads:MTLSizeMake(n,1,1)
       threadsPerThreadgroup:MTLSizeMake(std::min(n,(uint32_t)1024),1,1)];
    }
    { uint32_t n = (uint32_t)gi_q->size();
      auto enc = CommandBatch::get().encoder();
      [enc setComputePipelineState:add_k->getPipelineState()];
      [enc setBuffer:gi_q->getBuffer() offset:0 atIndex:0];
      [enc setBuffer:gi_v->getBuffer() offset:0 atIndex:1];
      [enc setBytes:&n length:4 atIndex:2];
      [enc dispatchThreads:MTLSizeMake(n,1,1)
       threadsPerThreadgroup:MTLSizeMake(std::min(n,(uint32_t)1024),1,1)];
    }

    dQ->invalidate(); dK->invalidate(); dV->invalidate();
    gi_k->invalidate(); gi_v->invalidate();
    return {gi_q};
}

void SparseAttention::update(Adafactor& optimizer) {
    for (auto& w : {w_q, w_k, w_v, w_o}) w->update(optimizer);
}
void SparseAttention::resparsify(float density) {
    for (auto& w : {w_q, w_k, w_v, w_o}) w->resparsify(density);
}
std::vector<std::shared_ptr<Tensor>> SparseAttention::parameters() {
    std::vector<std::shared_ptr<Tensor>> p;
    for (auto& w : {w_q, w_k, w_v, w_o}) {
        auto wp = w->parameters(); p.insert(p.end(), wp.begin(), wp.end()); }
    return p;
}
std::vector<std::shared_ptr<Tensor>> SparseAttention::param_gradients() {
    std::vector<std::shared_ptr<Tensor>> g;
    for (auto& w : {w_q, w_k, w_v, w_o}) {
        auto gp = w->param_gradients(); g.insert(g.end(), gp.begin(), gp.end()); }
    return g;
}
void SparseAttention::clear_gradients() {
    for (auto& w : {w_q, w_k, w_v, w_o}) w->clear_gradients();
}
void SparseAttention::clear_activations() {
    for (auto& w : {w_q, w_k, w_v, w_o}) w->clear_activations();
    last_input_.reset();
    saved_Q_.reset(); saved_K_.reset(); saved_V_.reset(); saved_attn_.reset();
}
void SparseAttention::save(std::ostream& os) const {
    w_q->save(os); w_k->save(os); w_v->save(os); w_o->save(os);
}
void SparseAttention::load(std::istream& is) {
    w_q->load(is); w_k->load(is); w_v->load(is); w_o->load(is);
}