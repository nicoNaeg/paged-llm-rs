// Paged attention for a decode step: one query token per row, reading a history
// that is scattered across physical blocks.
//
// One threadgroup per (row, head). The work splits three ways, and each split is
// chosen for how the reads land rather than for how the code reads.
//
// Scoring gives one position to each SIMD group and splits the head across its
// lanes, so the thirty-two lanes of a group read thirty-two consecutive floats
// of a key vector at a time and `simd_sum` finishes the dot product without a
// barrier. Softmax runs over the scores in threadgroup memory, in f32, which is
// what keeps a few thousand exponentials from losing their sum in bf16. The
// weighted sum of the value vectors gives one output dimension to each thread,
// so at every position the threads of a group read consecutive floats again.
//
// Nothing here materialises the history. The gather this replaces copied every
// resident sequence's keys and values into one rectangle before the multiply,
// per layer and per step, and the rectangle was as wide as the longest row.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint block_size;
    uint head_dim;
    uint num_heads;
    uint kv_heads;
    uint max_blocks;
    uint threads;
    uint simdgroups;
    float scale;
};

kernel void paged_attention_decode(
    device const ELEM*  q            [[buffer(0)]],
    device const ELEM*  k_pool       [[buffer(1)]],
    device const ELEM*  v_pool       [[buffer(2)]],
    device const uint*  block_table  [[buffer(3)]],
    device const uint*  context_lens [[buffer(4)]],
    device ELEM*        out          [[buffer(5)]],
    constant Params&    p            [[buffer(6)]],
    threadgroup float*  scores       [[threadgroup(0)]],
    threadgroup float*  scratch      [[threadgroup(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
    const uint head = tgid.x;
    const uint row = tgid.y;
    const uint ctx = context_lens[row];
    // Every thread of this group reads the same length, so they all leave
    // together and none is left waiting at a barrier the others skipped.
    if (ctx == 0) {
        return;
    }

    const uint group = p.num_heads / p.kv_heads;
    const uint kv_head = head / group;
    const uint kv_width = p.kv_heads * p.head_dim;
    device const ELEM* q_head = q + (row * p.num_heads + head) * p.head_dim;
    device const uint* table = block_table + row * p.max_blocks;

    // Scoring. The block table turns a logical position into a physical slot,
    // and that indirection is the whole of what paging costs the kernel.
    for (uint pos = sg; pos < ctx; pos += p.simdgroups) {
        const uint slot = table[pos / p.block_size] * p.block_size + pos % p.block_size;
        device const ELEM* k = k_pool + slot * kv_width + kv_head * p.head_dim;
        float acc = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32) {
            acc += float(q_head[d]) * float(k[d]);
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            scores[pos] = acc * p.scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax, shifted by the largest score so the exponential cannot overflow.
    float local = -INFINITY;
    for (uint i = tid; i < ctx; i += p.threads) {
        local = max(local, scores[i]);
    }
    scratch[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = p.threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float peak = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = 0.0f;
    for (uint i = tid; i < ctx; i += p.threads) {
        const float e = exp(scores[i] - peak);
        scores[i] = e;
        partial += e;
    }
    scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = p.threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float total = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The weighted sum. Thread d owns output dimension d, so the threads of a
    // group read consecutive floats of one value vector at each position.
    for (uint d = tid; d < p.head_dim; d += p.threads) {
        float acc = 0.0f;
        for (uint pos = 0; pos < ctx; ++pos) {
            const uint slot = table[pos / p.block_size] * p.block_size + pos % p.block_size;
            device const ELEM* v = v_pool + slot * kv_width + kv_head * p.head_dim;
            acc += scores[pos] * float(v[d]);
        }
        out[(row * p.num_heads + head) * p.head_dim + d] = ELEM(acc / total);
    }
}
