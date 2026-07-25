# paged-llm-rs

> LLM inference engine in Rust, served over the OpenAI API. Paged KV cache, continuous batching, attention kernels written in Metal Shading Language for Apple GPUs.

**Status: bootstrap.** The workspace, the lints, the CI and the device layer are in place. Nothing below the build order is built yet, and this README carries no performance number until a committed benchmark produces one.

## Design

An inference server's throughput is not decided by how fast its matrix multiplications are. Decoding one token reads every weight of the model out of memory, so a single sequence leaves the GPU's arithmetic units idle waiting on bandwidth. Batching is what fixes that: a second sequence rides along on the same weight read, and throughput climbs with the batch until memory runs out.

What runs out is the KV cache. Every token a sequence has seen leaves a key and a value vector per layer that later tokens attend to, and the usual implementation reserves one contiguous buffer per sequence, sized for the longest context it might reach. A request that stops after 100 tokens against a 4096 token reservation wastes 97 % of it, and that waste is exactly the memory a concurrent request needed.

PagedAttention removes the reservation. The cache becomes a pool of fixed-size blocks, each sequence holds a table mapping its logical positions to physical blocks, and blocks are handed out as tokens are produced. The attention kernel then reads keys and values that are scattered across the pool, through the table, instead of walking one contiguous buffer. This is virtual memory applied to attention, page table included, and it is the reason vLLM serves more concurrent requests than the memory arithmetic suggests it should.

The point here is to build that from its primitives. The block allocator, the scheduler that decides what runs on each step, the kernel that reads through a block table, the radix tree that lets two requests with a common prefix share physical blocks: all written in this repository, so the mechanics and their tradeoffs are the deliverable rather than the glue around an inference crate.

## Architecture

```mermaid
flowchart TB
    subgraph FE [Frontend]
        REQ[POST /v1/chat/completions] --> TOK[tokenize]
        TOK --> Q[waiting queue]
    end
    subgraph CP [Control plane]
        Q --> SCHED{scheduler, once per step}
        SCHED -->|memory available| ADM[admit: prefill]
        SCHED -->|already running| DEC[continue: decode]
        SCHED -->|pool exhausted| PRE[preempt, return blocks]
        ADM --> BM[block allocator]
        DEC --> BM
        BM --> BT[block tables]
    end
    subgraph DP [Data plane]
        BT --> FWD[one batched forward pass]
        FWD --> ATT[paged attention kernel, reads through the table]
        ATT --> SMP[sample]
    end
    SMP -->|token| SSE[stream to the client]
    SMP -->|sequence finished| FREE[free its blocks]
    SSE --> SCHED
```

The engine is synchronous and owns one model. The server runs on an async runtime and drives it from a blocking pool: a step is a blocking GPU dispatch whatever wraps it, and a synchronous engine stays testable and profilable without a runtime around it.

## What is written here and what is not

candle supplies tensors, safetensors loading and matrix multiplication on Metal. It is infrastructure: a serving engine is not a BLAS, and rewriting one would demonstrate nothing this project is about.

Everything above it is written here, including the model's forward pass. That last part is forced rather than chosen: `candle-transformers` ships Llama and Qwen implementations, and their attention keeps one contiguous KV cache per sequence, which is precisely the thing this project replaces. Writing RMSNorm, RoPE, grouped-query attention and the SwiGLU MLP against candle's primitives is what makes the attention layer able to call a paged kernel at all.

Every kernel ships with a CPU implementation. It is the oracle the GPU version is checked against, and it is also the only path CI can execute: `MTLCreateSystemDefaultDevice` returns nil inside GitHub's hosted macOS runners, so a hosted job compiles the Metal path and stops there.

## Build order

Each stage lands with its tests before the next one starts.

1. **Model forward pass** (planned): safetensors loading, RMSNorm, RoPE, grouped-query attention, SwiGLU, on candle primitives. Checked against a reference implementation's logits on the same weights.
2. **Server** (planned): OpenAI-compatible `/v1/completions` and `/v1/chat/completions`, streamed over server-sent events, one request at a time.
3. **Continuous batching, contiguous cache** (planned): the queue, the scheduler and the step loop, against a KV cache that is still one reserved buffer per sequence. This is the baseline the paged version has to beat, and it is measured before it is replaced.
4. **Block allocator** (planned): the paged layout as pure logic, no GPU involved, allocation, release, reference counting and block tables, fully unit tested.
5. **Paged attention kernel** (planned): the Metal kernel that reads keys and values through a block table, and its integration into the attention layer.
6. **Benchmarks and profiling** (planned): against `llama-server` and `mistral.rs` on the same machine with the same model, plus a GPU profile of the kernel. The delta between stages 3 and 5 is the headline result.
7. **Prefix caching** (planned): a radix tree over block hashes, so requests sharing a system prompt share physical blocks instead of recomputing them.
8. **Chunked prefill** (planned): a long prompt processed a slice per step, so an arriving request stops adding a latency spike to every sequence already decoding.

Quantized KV cache blocks and speculative decoding are not in the plan. They enter it if a measurement asks for them.

## Hardware and what it constrains

Development and measurement run on an Apple M4 Pro: 12 CPU cores, 16 GPU cores, 24 GB of unified memory at 273 GB/s.

Unified memory is what makes this different from a discrete GPU, and it cuts both ways. Weights do not have to be copied across a bus, but the block pool competes with the operating system for the same RAM, so sizing it is a real decision rather than a fraction of a dedicated VRAM budget. Bandwidth is the other constraint: 273 GB/s sets the ceiling on decode throughput before a single kernel is written.

Kernels are compiled from Metal Shading Language at startup rather than ahead of time, so building this repository needs no Xcode installation. Xcode is used for profiling with Instruments, which is a tool outside the crate rather than a dependency of it.

## Repository layout

    crates/pagedllm/         engine: model, scheduler, block allocator, kernels
    crates/pagedllm-server/  OpenAI-compatible HTTP server
    Makefile                 build, test and lint entry points

## Development

Requires a stable Rust toolchain; `rust-toolchain.toml` pins the channel and the components.

    make build       release build with the Metal backend
    make server      start the server
    make test        the CPU path, which is what CI runs
    make test-metal  adds the tests that need a Metal device
    make lint        rustfmt check, then clippy with warnings denied on both feature sets

## License

MIT
