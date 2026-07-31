# The CUDA backend, and what porting it said about the Metal one

The paged attention kernel runs on NVIDIA as well as on Apple. It was written as
a port of the Metal one rather than a transcription of vLLM's, compiled and
checked on a rented RTX 4090, and the interesting part is not that it works. It
is what the second backend says about the first.

    cargo test --features cuda        # 96 tests, kernel against its oracle
    cargo run --release --features cuda --bin pagedllm-server -- --model ...

Measured on an RTX 4090, driver 525.105.17, CUDA 11.8, Qwen3-0.6B in bfloat16,
against the same commands the Metal numbers come from.

## The kernel agrees with the oracle

Worst difference against the tensor path on the GPU: **5.960e-8**, which is f32
rounding, and the same order as Metal's 1.0e-7. The scalar reference that CI runs
is unchanged and is the oracle for both, which is the whole reason it exists. All
96 tests pass with `--features cuda`, including the one that compiles every dtype
through NVRTC.

## What the port cost, line by line

| Metal | CUDA |
|---|---|
| one threadgroup per (row, head) | one block per (row, head), `gridDim = (heads, rows)` |
| SIMD group of 32 lanes | warp of 32 lanes |
| `simd_sum(acc)` | `__shfl_down_sync` folded over 16, 8, 4, 2, 1 |
| `threadgroup_barrier(mem_threadgroup)` | `__syncthreads()` |
| two bound threadgroup allocations | one `extern __shared__`, partitioned by hand |
| `newLibraryWithSource` at startup | NVRTC at startup, include path from `CUDA_PATH` |

Three things only compiling could have found, and all three are in the first
paragraph of what a port owes: NVRTC ships no headers, so `cuda_fp16.h` has to be
pointed at; it defines no `INFINITY`, so the softmax takes its starting maximum
from `math_constants.h`; and `CudaSlice` borrows its storage, so the guards have
to outlive the launch rather than being handed back from a helper.

This is also where `unsafe_code = "deny"` is finally opted out of, and the
decision log had predicted the wrong backend. Metal needed none, because candle
wraps every step. CUDA needs exactly one, `LaunchArgs::launch`, and what the
compiler stops checking there is that the kernel's parameter list matches the
arguments pushed into the builder. Nothing but running it against the oracle can
check that.

## The finding: the kernel's 4.1x was a statement about candle's Metal backend

Five configurations, one invocation, 3584 MiB of cache, `make bench-concurrency`,
output tokens a second:

| clients | reservation | paging | paging and kernel | paging, sliced | paging and kernel, sliced |
|---|---|---|---|---|---|
| 1 | 73.4 | 73.8 | **88.7** | 73.6 | 88.1 |
| 4 | 251.3 | 250.8 | 294.0 | 247.5 | 297.7 |
| 16 | 583.8 | 672.4 | 738.8 | 675.9 | 766.7 |
| 32 | 683.8 | 943.9 | 970.7 | 977.7 | **1153.5** |
| 64 | 686.7 | 1013.9 | 995.7 | 1044.7 | **1108.0** |
| ttft at 64 | 3216 ms | 474 ms | 484 ms | 682 ms | 597 ms |

**On CUDA the hand-written kernel buys 21 % at one client and nothing at
sixty-four**, where on Metal it bought 4.1 times at sixty-four. At 64 clients it
is 995.7 against the tensor gather's 1013.9, which is behind it. Two runs agreed
to within a few percent, so this is not noise.

The Metal number was never wrong, and it is what makes this one worth publishing:
the 4.1x was measuring candle's Metal gather, not paged attention. On CUDA the
same gather is already good, so the same hand-written kernel has nothing left to
take. A kernel is faster than the thing it replaces, and how much depends
entirely on what that thing was.

Paging still pays, for the reason it always did. A reservation caps concurrency:
686.7 tok/s and 3216 ms to a first token at 64 clients, against 1013.9 and 474.
That is 1.5x the throughput and 6.8x the first token, where Metal measured 3.0x
and 39x. The direction is the same on both and the size is not.

## The pass budget's right value is hardware dependent

`make bench-chunk`, worst gap between two tokens for clients already streaming:

| `--chunk` | worst gap | newcomer's first token |
|---|---|---|
| off | 151 ms | 95 ms |
| 512 | 50 ms | 102 ms |
| 128 (the default) | 25 ms | 149 ms |
| 64 | 16 ms | 205 ms |

The default of 128 was measured on Metal, where it cost the arriving prompt 4 %.
Here it costs 57 %, and 512 buys 3.0x for 7 %. Nothing is broken; the crossover
simply sits somewhere else on a machine whose prefill is an order of magnitude
faster. A constant picked on one machine is a constant picked on one machine,
which is why it is a flag.

Prefix caching moves the same way: `make bench-prefix` gives 3.26x on the first
token here against 7.2x on Metal, and 1.06x where nothing is shared against 1.10,
so it still costs nothing where it can buy nothing.

## What CI can and cannot say about this

Nothing. A hosted runner has no NVIDIA device, which is the same as the Metal
story, but it is worse in one way: `candle-core/cuda` needs `nvcc` to build at
all, so CI cannot even compile this path where it does compile the Metal one.
What holds it up is the scalar reference, which CI does run and which is the
oracle both kernels are checked against, plus a rented hour whenever the kernel
changes. That is a weaker guarantee than the rest of this repository has, and it
is stated here rather than left to be discovered.

## The comparison that is still missing

vLLM does not run on the box this was measured on. Its bundled PyTorch is built
against a newer CUDA than the host driver offers (525.105.17, which is CUDA
12.0), and it fails at `torch._C._cuda_init` before the engine starts. Installing
an older vLLM built against cu121 failed on that machine's network rather than on
anything about the code. So the two-paged-implementations comparison this port
was worth paying for still has not happened, and the honest statement is that it
is one rented hour away rather than that it was made.

This engine itself has no such problem on that driver: candle compiles against
the CUDA 11.8 toolkit the image carries, which is why every number above exists.
