// Paged attention for a decode step, on CUDA. A port of the Metal kernel beside
// it, and deliberately the same design rather than a transcription of vLLM's.
//
// The mapping is one to one, which is the point: a threadgroup is a block, a
// SIMD group is a warp, `simd_sum` is a shuffle reduction, and threadgroup
// memory is `__shared__`. What does not map is the shared memory declaration:
// Metal takes one length per bound index, CUDA gives one dynamic block, so the
// scores and the reduction scratch are carved out of it here rather than bound
// separately.
//
// One block per (row, head). Scoring gives one position to each warp and splits
// the head across its thirty-two lanes, so a warp reads thirty-two consecutive
// values of a key vector at a time and the shuffle finishes the dot product with
// no barrier. Softmax runs over the scores in shared memory in f32, which is
// what keeps a few thousand exponentials from losing their sum in bf16. The
// weighted sum gives one output dimension to each thread, so the threads of a
// warp again read consecutive values.
//
// Nothing here materialises the history: the block table turns a logical
// position into a physical slot and the kernel reads the slot where it is.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
// NVRTC compiles without the host's math headers, so the infinity the softmax
// starts its maximum from comes from the toolkit's own constants rather than
// from <math.h>.
#include <math_constants.h>

// Reading an element as f32 whatever it is stored as. Metal casts with a
// constructor; here the half types need their own conversion, so the widening
// goes through an overload set rather than a cast.
__device__ __forceinline__ float widen(float x) { return x; }
__device__ __forceinline__ float widen(__half x) { return __half2float(x); }
__device__ __forceinline__ float widen(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ void narrow_to(float *out, float x) { *out = x; }
__device__ __forceinline__ void narrow_to(__half *out, float x) { *out = __float2half(x); }
__device__ __forceinline__ void narrow_to(__nv_bfloat16 *out, float x) {
    *out = __float2bfloat16(x);
}

extern "C" __global__ void paged_attention_decode(
    const ELEM *__restrict__ q,
    const ELEM *__restrict__ k_pool,
    const ELEM *__restrict__ v_pool,
    const unsigned int *__restrict__ block_table,
    const unsigned int *__restrict__ context_lens,
    ELEM *__restrict__ out,
    const unsigned int block_size,
    const unsigned int head_dim,
    const unsigned int num_heads,
    const unsigned int kv_heads,
    const unsigned int max_blocks,
    const unsigned int threads,
    const unsigned int warps,
    const float scale)
{
    const unsigned int head = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int ctx = context_lens[row];
    // Every thread of this block reads the same length, so they all leave
    // together and none is left waiting at a barrier the others skipped.
    if (ctx == 0) {
        return;
    }

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32u;
    const unsigned int warp = tid / 32u;

    // One dynamic allocation, split by hand: the scores first, one float per
    // position the longest row reaches, then one float per thread for the two
    // tree reductions below.
    extern __shared__ float shared[];
    float *scores = shared;
    float *scratch = shared + max_blocks * block_size;

    const unsigned int group = num_heads / kv_heads;
    const unsigned int kv_head = head / group;
    const unsigned int kv_width = kv_heads * head_dim;
    const ELEM *q_head = q + (row * num_heads + head) * head_dim;
    const unsigned int *table = block_table + row * max_blocks;

    // Scoring. The block table turns a logical position into a physical slot,
    // and that indirection is the whole of what paging costs the kernel.
    for (unsigned int pos = warp; pos < ctx; pos += warps) {
        const unsigned int slot =
            table[pos / block_size] * block_size + pos % block_size;
        const ELEM *k = k_pool + slot * kv_width + kv_head * head_dim;
        float acc = 0.0f;
        for (unsigned int d = lane; d < head_dim; d += 32u) {
            acc += widen(q_head[d]) * widen(k[d]);
        }
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0u) {
            scores[pos] = acc * scale;
        }
    }
    __syncthreads();

    // Softmax, shifted by the largest score so the exponential cannot overflow.
    float local = -CUDART_INF_F;
    for (unsigned int i = tid; i < ctx; i += threads) {
        local = fmaxf(local, scores[i]);
    }
    scratch[tid] = local;
    __syncthreads();
    for (unsigned int stride = threads / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    const float peak = scratch[0];
    __syncthreads();

    float partial = 0.0f;
    for (unsigned int i = tid; i < ctx; i += threads) {
        const float e = expf(scores[i] - peak);
        scores[i] = e;
        partial += e;
    }
    scratch[tid] = partial;
    __syncthreads();
    for (unsigned int stride = threads / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();

    // The weighted sum. Thread d owns output dimension d, so the threads of a
    // warp read consecutive values of one value vector at each position.
    for (unsigned int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (unsigned int pos = 0; pos < ctx; ++pos) {
            const unsigned int slot =
                table[pos / block_size] * block_size + pos % block_size;
            const ELEM *v = v_pool + slot * kv_width + kv_head * head_dim;
            acc += scores[pos] * widen(v[d]);
        }
        narrow_to(&out[(row * num_heads + head) * head_dim + d], acc / total);
    }
}
