#include <metal_stdlib>
using namespace metal;

inline float bf16_to_fp32(ushort v) { return as_type<float>((uint)v << 16); }
inline ushort fp32_to_bf16(float v) {
    uint b = as_type<uint>(v);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (ushort)(b >> 16);
}

kernel void fill_bf16(
    device ushort  *out  [[ buffer(0) ]],
    constant float &val  [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]])
{
    uint b = as_type<uint>(val);
    b += 0x7FFFu + ((b >> 16) & 1u);
    out[id] = (ushort)(b >> 16);
}

kernel void elementwise_add(
    device float *a [[ buffer(0) ]], device const float *b [[ buffer(1) ]],
    constant uint &n [[ buffer(2) ]], uint id [[ thread_position_in_grid ]])
{
    if (id >= n) return;
    a[id] += b[id];
}

kernel void elementwise_add3(
    device float *out [[ buffer(0) ]], device const float *a [[ buffer(1) ]],
    device const float *b [[ buffer(2) ]], constant uint &n [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= n) return;
    out[id] = a[id] + b[id];
}

kernel void elementwise_mul(
    device float *out [[ buffer(0) ]], device const float *a [[ buffer(1) ]],
    device const float *b [[ buffer(2) ]], constant uint &n [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= n) return;
    out[id] = a[id] * b[id];
}


kernel void silu_forward(
    device float *out [[ buffer(0) ]], device const float *inp [[ buffer(1) ]],
    constant uint &n [[ buffer(2) ]], uint id [[ thread_position_in_grid ]])
{
    if (id >= n) return;
    float x = inp[id];
    out[id] = x / (1.0f + exp(-x));
}

kernel void silu_backward(
    device float *dx [[ buffer(0) ]], device const float *dy [[ buffer(1) ]],
    device const float *x [[ buffer(2) ]], constant uint &n [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= n) return;
    float xv = x[id];
    float sig = 1.0f / (1.0f + exp(-xv));
    dx[id] = dy[id] * sig * (1.0f + xv * (1.0f - sig));
}


kernel void rms_norm_forward(
    device float *out [[ buffer(0) ]], device const float *inp [[ buffer(1) ]],
    constant uint &dim [[ buffer(2) ]], constant float &eps [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    uint base = id * dim;
    float ss = 0.0f;
    for (uint i = 0; i < dim; ++i) { float v = inp[base+i]; ss += v*v; }
    float scale = rsqrt(ss / (float)dim + eps);
    for (uint i = 0; i < dim; ++i) out[base+i] = inp[base+i] * scale;
}

kernel void rms_norm_backward(
    device float *dx [[ buffer(0) ]], device const float *dy [[ buffer(1) ]],
    device const float *x_orig [[ buffer(2) ]], constant uint &dim [[ buffer(3) ]],
    constant float &eps [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]])
{
    uint base = id * dim;
    float ss = 0.0f;
    for (uint i = 0; i < dim; ++i) { float v = x_orig[base+i]; ss += v*v; }
    float scale = rsqrt(ss / (float)dim + eps);
    float dot = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        float y = x_orig[base+i] * scale;
        dot += dy[base+i] * y;
    }
    float inv_n = 1.0f / (float)dim;
    for (uint i = 0; i < dim; ++i) {
        float y = x_orig[base+i] * scale;
        dx[base+i] = scale * (dy[base+i] - y * dot * inv_n);
    }
}


kernel void cross_entropy_loss(
    device float *ls [[ buffer(0) ]], device float *gs [[ buffer(1) ]],
    device const float *l [[ buffer(2) ]], device const int *t [[ buffer(3) ]],
    constant uint &b [[ buffer(4) ]], constant uint &v [[ buffer(5) ]],
    uint bid [[ threadgroup_position_in_grid ]], uint tid [[ thread_position_in_threadgroup ]],
    uint tgs [[ threads_per_threadgroup ]])
{
    if (bid >= b) return;
    device const float* logit = l + bid * v;
    device float* grad = gs + bid * v;
    threadgroup float shared[1024];
    float lmax = -1e9f;
    for (uint i = tid; i < v; i += tgs) lmax = max(lmax, logit[i]);
    shared[tid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid+s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mx = shared[0];
    float lsum = 0.f;
    for (uint i = tid; i < v; i += tgs) lsum += exp(logit[i] - mx);
    shared[tid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float se = shared[0];
    if (tid == 0) ls[bid] = -(logit[t[bid]] - mx - log(se));
    float inv_b = 1.f / (float)b;
    for (uint i = tid; i < v; i += tgs)
        grad[i] = (exp(logit[i]-mx)/se - (i==(uint)t[bid] ? 1.f : 0.f)) * inv_b;
}


struct SparseParams { uint batch_size; uint in_features; uint out_features; float output_scale; };

kernel void sparse_ternary_matmul_free(
    device float        *out          [[ buffer(0) ]],
    device const float  *input        [[ buffer(1) ]],
    device const uint   *pos          [[ buffer(2) ]],
    device const uint   *neg          [[ buffer(3) ]],
    device const uint2  *ptrs         [[ buffer(4) ]],
    device const uint2  *counts       [[ buffer(5) ]],
    device const ushort *packed_pos_w [[ buffer(6) ]],
    device const ushort *packed_neg_w [[ buffer(7) ]],
    constant SparseParams &p          [[ buffer(8) ]],
    uint2 id [[ thread_position_in_grid ]])
{
    if (id.x >= p.out_features || id.y >= p.batch_size) return;
    float sum = 0.0f;
    uint off = id.y * p.in_features;
    uint row = id.x;
    uint pos_base = ptrs[row].x;
    uint neg_base = ptrs[row].y;
    for (uint i = 0; i < counts[row].x; ++i)
        sum += bf16_to_fp32(packed_pos_w[pos_base + i]) * input[off + pos[pos_base + i]];
    for (uint i = 0; i < counts[row].y; ++i)
        sum += bf16_to_fp32(packed_neg_w[neg_base + i]) * input[off + neg[neg_base + i]];
    out[id.y * p.out_features + id.x] = sum * p.output_scale;
}

kernel void sparse_ternary_backward_input(
    device float        *gi           [[ buffer(0) ]],
    device const float  *go           [[ buffer(1) ]],
    device const uint   *pos          [[ buffer(2) ]],
    device const uint   *neg          [[ buffer(3) ]],
    device const uint2  *ptrs         [[ buffer(4) ]],
    device const uint2  *counts       [[ buffer(5) ]],
    device const ushort *packed_pos_w [[ buffer(6) ]],
    device const ushort *packed_neg_w [[ buffer(7) ]],
    constant SparseParams &p          [[ buffer(8) ]],
    uint2 id  [[ thread_position_in_grid ]],
    uint2 tid [[ thread_position_in_threadgroup ]],
    uint2 tgs [[ threads_per_threadgroup ]])
{
    if (id.x >= p.in_features || id.y >= p.batch_size) return;
    uint col = id.x, batch = id.y;
    uint pos_base = ptrs[col].x, neg_base = ptrs[col].y;
    uint pos_cnt = counts[col].x, neg_cnt = counts[col].y;
    uint nnz_tid = tid.y, nnz_tgs = tgs.y;
    float sum = 0.f;
    uint off = batch * p.out_features;
    for (uint i = nnz_tid; i < pos_cnt; i += nnz_tgs)
        sum += bf16_to_fp32(packed_pos_w[pos_base + i]) * go[off + pos[pos_base + i]];
    for (uint i = nnz_tid; i < neg_cnt; i += nnz_tgs)
        sum += bf16_to_fp32(packed_neg_w[neg_base + i]) * go[off + neg[neg_base + i]];
    threadgroup float shared[32];
    shared[nnz_tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (nnz_tid == 0) {
        float total = 0.f;
        for (uint t = 0; t < nnz_tgs; ++t) total += shared[t];
        gi[batch * p.in_features + col] = total * p.output_scale;
    }
}


struct SparseWeightBwdParams { uint batch_size, in_features, out_features, nnz; float output_scale; };

kernel void sparse_backward_weight_pos(
    device float       *gw_pos  [[ buffer(0) ]],
    device const float *go      [[ buffer(1) ]],
    device const float *inp     [[ buffer(2) ]],
    device const uint  *row_idx [[ buffer(3) ]],
    device const uint  *col_idx [[ buffer(4) ]],
    constant SparseWeightBwdParams &p [[ buffer(5) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= p.nnz) return;
    float acc = 0.0f;
    for (uint b = 0; b < p.batch_size; ++b)
        acc += go[b * p.out_features + row_idx[id]] * inp[b * p.in_features + col_idx[id]];
    gw_pos[id] = acc * p.output_scale;
}

kernel void sparse_backward_weight_neg(
    device float       *gw_neg  [[ buffer(0) ]],
    device const float *go      [[ buffer(1) ]],
    device const float *inp     [[ buffer(2) ]],
    device const uint  *row_idx [[ buffer(3) ]],
    device const uint  *col_idx [[ buffer(4) ]],
    constant SparseWeightBwdParams &p [[ buffer(5) ]],
    uint id [[ thread_position_in_grid ]])
{
    if (id >= p.nnz) return;
    float acc = 0.0f;
    for (uint b = 0; b < p.batch_size; ++b)
        acc += go[b * p.out_features + row_idx[id]] * inp[b * p.in_features + col_idx[id]];
    gw_neg[id] = acc * p.output_scale;
}


kernel void dense_bf16_matmul(
    device float *out [[ buffer(0) ]], device const float *x [[ buffer(1) ]],
    device const ushort *W [[ buffer(2) ]], constant uint &BS [[ buffer(3) ]],
    constant uint &dim [[ buffer(4) ]], constant uint &vocab [[ buffer(5) ]],
    uint2 id [[ thread_position_in_grid ]])
{
    uint v = id.x, b = id.y;
    if (v >= vocab || b >= BS) return;
    float sum = 0.f;
    for (uint d = 0; d < dim; ++d) sum += x[b*dim+d] * bf16_to_fp32(W[d*vocab+v]);
    out[b*vocab+v] = sum;
}

kernel void dense_bf16_matmul_bwd_x(
    device float *gx [[ buffer(0) ]], device const float *glogits [[ buffer(1) ]],
    device const ushort *W [[ buffer(2) ]], constant uint &BS [[ buffer(3) ]],
    constant uint &dim [[ buffer(4) ]], constant uint &vocab [[ buffer(5) ]],
    uint2 id [[ thread_position_in_grid ]])
{
    uint d = id.x, b = id.y;
    if (d >= dim || b >= BS) return;
    float sum = 0.f;
    for (uint v = 0; v < vocab; ++v) sum += glogits[b*vocab+v] * bf16_to_fp32(W[d*vocab+v]);
    gx[b*dim+d] = sum;
}

kernel void dense_bf16_matmul_bwd_w(
    device float *gW [[ buffer(0) ]], device const float *x [[ buffer(1) ]],
    device const float *glogits [[ buffer(2) ]], constant uint &BS [[ buffer(3) ]],
    constant uint &dim [[ buffer(4) ]], constant uint &vocab [[ buffer(5) ]],
    uint2 id [[ thread_position_in_grid ]])
{
    uint d = id.x, v = id.y;
    if (d >= dim || v >= vocab) return;
    float sum = 0.f;
    for (uint b = 0; b < BS; ++b) sum += x[b*dim+d] * glogits[b*vocab+v];
    gW[d*vocab+v] = sum;
}

// ─── Adafactor ────────────────────────────────────────────────────────────────

struct AdafactorParams { float lr, decay, e1, e2, clip; uint b, i, o; int step; };

kernel void adafactor_reduce(
    device float *rv [[ buffer(0) ]], device const float *g [[ buffer(1) ]],
    constant AdafactorParams &p [[ buffer(2) ]], uint id [[ thread_position_in_grid ]])
{
    if (id >= p.o) return;
    float sum = 0.0f;
    for (uint j = 0; j < p.i; ++j) { float v = g[id * p.i + j]; sum += v * v; }
    rv[id] = p.decay * rv[id] + (1.0f - p.decay) * (sum / (float)p.i);
}

kernel void adafactor_reduce_col(
    device float *cv [[ buffer(0) ]], device const float *g [[ buffer(1) ]],
    constant AdafactorParams &p [[ buffer(2) ]], uint id [[ thread_position_in_grid ]])
{
    if (id >= p.i) return;
    float sum = 0.0f;
    for (uint i = 0; i < p.o; ++i) { float v = g[i * p.i + id]; sum += v * v; }
    cv[id] = p.decay * cv[id] + (1.0f - p.decay) * (sum / (float)p.o);
}

kernel void adafactor_update(
    device float *w [[ buffer(0) ]], device const float *g [[ buffer(1) ]],
    device const float *rv [[ buffer(2) ]], device const float *cv [[ buffer(3) ]],
    constant AdafactorParams &p [[ buffer(4) ]], uint2 id [[ thread_position_in_grid ]])
{
    if (id.x >= p.i || id.y >= p.o) return;
    uint idx = id.y * p.i + id.x;
    w[idx] -= p.lr * g[idx] * rsqrt(rv[id.y] * cv[id.x] + p.e2);
}

kernel void adafactor_reduce_bf16(
    device ushort *rv [[ buffer(0) ]], device const float *g [[ buffer(1) ]],
    constant AdafactorParams &p [[ buffer(2) ]], uint id [[ thread_position_in_grid ]])
{
    if (id >= p.o) return;
    float sum = 0.0f;
    for (uint j = 0; j < p.i; ++j) { float v = g[id * p.i + j]; sum += v * v; }
    float nv = p.decay * bf16_to_fp32(rv[id]) + (1.0f - p.decay) * (sum / (float)p.i);
    rv[id] = fp32_to_bf16(nv);
}

kernel void adafactor_reduce_col_bf16(
    device ushort *cv [[ buffer(0) ]], device const float *g [[ buffer(1) ]],
    constant AdafactorParams &p [[ buffer(2) ]], uint id [[ thread_position_in_grid ]])
{
    if (id >= p.i) return;
    float sum = 0.0f;
    for (uint i = 0; i < p.o; ++i) { float v = g[i * p.i + id]; sum += v * v; }
    float nv = p.decay * bf16_to_fp32(cv[id]) + (1.0f - p.decay) * (sum / (float)p.o);
    cv[id] = fp32_to_bf16(nv);
}

kernel void adafactor_update_bf16(
    device ushort *w [[ buffer(0) ]], device const float *g [[ buffer(1) ]],
    device const ushort *rv [[ buffer(2) ]], device const ushort *cv [[ buffer(3) ]],
    constant AdafactorParams &p [[ buffer(4) ]], uint2 id [[ thread_position_in_grid ]])
{
    if (id.x >= p.i || id.y >= p.o) return;
    uint idx = id.y * p.i + id.x;
    float fw = bf16_to_fp32(w[idx]);
    fw -= p.lr * g[idx] * rsqrt(bf16_to_fp32(rv[id.y]) * bf16_to_fp32(cv[id.x]) + p.e2);
    w[idx] = fp32_to_bf16(fw);
}


struct FusedAdamParams {
    float scale, clip_scale, lr, beta1, beta2, eps, bc1, bc2;
    uint32_t nnz, in_features;
};

kernel void fused_scale_adam_sparse_bf16(
    device ushort *w [[ buffer(0) ]], device const float *ng [[ buffer(1) ]],
    device float *m_buf [[ buffer(2) ]], device float *v_buf [[ buffer(3) ]],
    constant FusedAdamParams &p [[ buffer(4) ]], uint id [[ thread_position_in_grid ]])
{
    if (id >= p.nnz) return;
    float g = ng[id] * p.scale * p.clip_scale;
    float m = p.beta1 * m_buf[id] + (1.f - p.beta1) * g;
    float v = p.beta2 * v_buf[id] + (1.f - p.beta2) * g * g;
    m_buf[id] = m; v_buf[id] = v;
    float step = p.lr * (m / p.bc1) / (sqrt(v / p.bc2) + p.eps);
    float fw = as_type<float>((uint32_t)w[id] << 16);
    fw -= step;
    uint32_t b32 = as_type<uint32_t>(fw);
    b32 += 0x7FFFu + ((b32 >> 16) & 1u);
    w[id] = (ushort)(b32 >> 16);
}


struct CausalAttnParams { uint bh, S, hd; float scale; uint window; };

kernel void causal_qk_scores(
    device float       *scores [[ buffer(0) ]],
    device const float *Q      [[ buffer(1) ]],
    device const float *K      [[ buffer(2) ]],
    constant CausalAttnParams &p [[ buffer(3) ]],
    uint3 id [[ thread_position_in_grid ]])
{
    uint qi = id.x, ki = id.y, h = id.z;
    if (qi >= p.S || ki >= p.S || h >= p.bh) return;
    if (ki > qi) { scores[h * p.S * p.S + qi * p.S + ki] = -1e9f; return; }
    if (p.window > 0 && (qi - ki) > p.window) { scores[h * p.S * p.S + qi * p.S + ki] = -1e9f; return; }
    float sum = 0.f;
    uint qbase = h * p.S * p.hd + qi * p.hd;
    uint kbase = h * p.S * p.hd + ki * p.hd;
    for (uint d = 0; d < p.hd; ++d) sum += Q[qbase + d] * K[kbase + d];
    scores[h * p.S * p.S + qi * p.S + ki] = sum * p.scale;
}

kernel void row_softmax(
    device float *scores [[ buffer(0) ]],
    constant CausalAttnParams &p [[ buffer(1) ]],
    uint2 id [[ thread_position_in_grid ]])
{
    uint qi = id.x, h = id.y;
    if (qi >= p.S || h >= p.bh) return;
    uint base = h * p.S * p.S + qi * p.S;
    float mx = -1e9f;
    for (uint k = 0; k < p.S; ++k) mx = max(mx, scores[base + k]);
    float sum = 0.f;
    for (uint k = 0; k < p.S; ++k) {
        float e = exp(scores[base + k] - mx);
        scores[base + k] = e;
        sum += e;
    }
    float inv = 1.0f / (sum + 1e-9f);
    for (uint k = 0; k < p.S; ++k) scores[base + k] *= inv;
}

kernel void attn_weighted_sum(
    device float       *out   [[ buffer(0) ]],
    device const float *attn  [[ buffer(1) ]],
    device const float *V     [[ buffer(2) ]],
    constant CausalAttnParams &p [[ buffer(3) ]],
    uint3 id [[ thread_position_in_grid ]])
{
    uint d = id.x, qi = id.y, h = id.z;
    if (d >= p.hd || qi >= p.S || h >= p.bh) return;
    float sum = 0.f;
    uint abase = h * p.S * p.S + qi * p.S;
    for (uint ki = 0; ki < p.S; ++ki)
        sum += attn[abase + ki] * V[h * p.S * p.hd + ki * p.hd + d];
    out[h * p.S * p.hd + qi * p.hd + d] = sum;
}

kernel void attn_bwd_dv(
    device float       *dV   [[ buffer(0) ]],
    device const float *attn [[ buffer(1) ]],
    device const float *dOut [[ buffer(2) ]],
    constant CausalAttnParams &p [[ buffer(3) ]],
    uint3 id [[ thread_position_in_grid ]])
{
    uint d = id.x, ki = id.y, h = id.z;
    if (d >= p.hd || ki >= p.S || h >= p.bh) return;
    float sum = 0.f;
    for (uint qi = 0; qi < p.S; ++qi)
        sum += attn[h * p.S * p.S + qi * p.S + ki] * dOut[h * p.S * p.hd + qi * p.hd + d];
    dV[h * p.S * p.hd + ki * p.hd + d] = sum;
}

kernel void attn_bwd_dattn(
    device float       *da   [[ buffer(0) ]],
    device const float *dOut [[ buffer(1) ]],
    device const float *V    [[ buffer(2) ]],
    constant CausalAttnParams &p [[ buffer(3) ]],
    uint3 id [[ thread_position_in_grid ]])
{
    uint qi = id.x, ki = id.y, h = id.z;
    if (qi >= p.S || ki >= p.S || h >= p.bh) return;
    float sum = 0.f;
    uint obase = h * p.S * p.hd + qi * p.hd;
    uint vbase = h * p.S * p.hd + ki * p.hd;
    for (uint d = 0; d < p.hd; ++d) sum += dOut[obase + d] * V[vbase + d];
    da[h * p.S * p.S + qi * p.S + ki] = sum;
}

kernel void attn_bwd_softmax(
    device float       *da   [[ buffer(0) ]],
    device const float *attn [[ buffer(1) ]],
    constant CausalAttnParams &p [[ buffer(2) ]],
    uint2 id [[ thread_position_in_grid ]])
{
    uint qi = id.x, h = id.y;
    if (qi >= p.S || h >= p.bh) return;
    uint base = h * p.S * p.S + qi * p.S;
    float dot = 0.f;
    for (uint ki = 0; ki < p.S; ++ki) dot += attn[base + ki] * da[base + ki];
    for (uint ki = 0; ki < p.S; ++ki)
        da[base + ki] = attn[base + ki] * (da[base + ki] - dot) * p.scale;
}

kernel void attn_bwd_dq_matmul(
    device float       *dQ   [[ buffer(0) ]],
    device const float *ds   [[ buffer(1) ]],
    device const float *K    [[ buffer(2) ]],
    constant CausalAttnParams &p [[ buffer(3) ]],
    uint3 id [[ thread_position_in_grid ]])
{
    uint d = id.x, qi = id.y, h = id.z;
    if (d >= p.hd || qi >= p.S || h >= p.bh) return;
    float sum = 0.f;
    uint sbase = h * p.S * p.S + qi * p.S;
    for (uint ki = 0; ki < p.S; ++ki)
        sum += ds[sbase + ki] * K[h * p.S * p.hd + ki * p.hd + d];
    dQ[h * p.S * p.hd + qi * p.hd + d] = sum;
}

kernel void attn_bwd_dk_matmul(
    device float       *dK   [[ buffer(0) ]],
    device const float *ds   [[ buffer(1) ]],
    device const float *Q    [[ buffer(2) ]],
    constant CausalAttnParams &p [[ buffer(3) ]],
    uint3 id [[ thread_position_in_grid ]])
{
    uint d = id.x, ki = id.y, h = id.z;
    if (d >= p.hd || ki >= p.S || h >= p.bh) return;
    float sum = 0.f;
    for (uint qi = 0; qi < p.S; ++qi)
        sum += ds[h * p.S * p.S + qi * p.S + ki] * Q[h * p.S * p.hd + qi * p.hd + d];
    dK[h * p.S * p.hd + ki * p.hd + d] = sum;
}
