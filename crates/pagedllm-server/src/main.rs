//! Entry point for the inference server.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, ValueEnum};
use pagedllm::{Backend, DType, GenerationConfig};
use pagedllm_server::{AppState, Engine, PoolConfig, router};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Dtype {
    /// Whatever the checkpoint stores.
    Auto,
    F32,
    Bf16,
}

#[derive(Debug, Parser)]
#[command(about = "OpenAI-compatible inference server", version)]
struct Args {
    /// Directory holding config.json, model.safetensors and the tokenizer.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Convert the weights on load. `auto` keeps the checkpoint's own dtype,
    /// which the CPU backend cannot run for bf16, since candle has no bf16
    /// matmul there.
    #[arg(long, value_enum, default_value_t = Dtype::Auto)]
    dtype: Dtype,
    /// Refuse to start if the Metal backend was compiled in but is unreachable.
    #[arg(long)]
    require_metal: bool,
    /// Token budget for a request that names none.
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    /// Tokens per cache block. Set this as wide as the context and every
    /// sequence gets exactly one block, which is a reservation per sequence and
    /// the layout stage 3 measured. That is how the two are compared.
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    /// Bytes the KV cache may take, in mebibytes. The block count follows from
    /// it and from the model's shape, and both are printed at startup.
    #[arg(long, default_value_t = 3584)]
    cache_mib: usize,
    /// Rows in one decode pass, capped separately from the slots so a batch can
    /// be limited without shrinking the cache.
    #[arg(long, default_value_t = 16)]
    max_batch: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let backend = Backend::detect();
    if args.require_metal && backend == Backend::Cpu {
        return Err("no Metal device, and --require-metal was given".into());
    }
    let device = backend.device()?;
    let dtype = match args.dtype {
        Dtype::Auto => None,
        Dtype::F32 => Some(DType::F32),
        Dtype::Bf16 => Some(DType::BF16),
    };

    let generation = Arc::new(GenerationConfig::from_dir(&args.model)?);
    // The block count comes from the budget rather than the other way round, so
    // changing the block size compares two layouts at the same memory.
    let shape = pagedllm::Config::from_file(args.model.join("config.json"))?;
    let per_token = pagedllm::CacheConfig {
        block_size: args.block_size,
        blocks: 1,
        kv_heads: shape.num_key_value_heads,
        head_dim: shape.head_dim,
        layers: shape.num_hidden_layers,
    };
    let budget = args.cache_mib << 20;
    let dtype_for_pool = dtype.unwrap_or(pagedllm::DType::BF16);
    let pool = PoolConfig {
        block_size: args.block_size,
        blocks: per_token.blocks_within(budget, dtype_for_pool),
        max_batch: args.max_batch,
    };
    let engine = Engine::start(args.model.clone(), device.clone(), dtype, pool)
        .await
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Printed rather than assumed. A build compiled with --features metal that
    // fell back to the CPU would make every later measurement a lie, and the
    // sampling defaults decide what the model actually answers.
    println!("pagedllm-server {}", env!("CARGO_PKG_VERSION"));
    println!("  model    {}", args.model.display());
    println!("  backend  {backend}");
    println!(
        "  sampling temperature {:?}, top_p {:?}, top_k {:?}, from the model's generation config",
        generation.temperature, generation.top_p, generation.top_k
    );
    println!("  stop     {:?}", generation.eos_token_ids);
    // Printed because it is the number stage 5 has to raise. A slot costs its
    // whole reservation the moment a sequence takes it, so this is the concurrency
    // ceiling, decided before any request arrives.
    println!(
        "  cache    {} blocks of {} tokens, {:.2} GiB, {} bytes a token, {} tokens held at once",
        pool.blocks,
        pool.block_size,
        engine.cache_bytes() as f64 / (1u64 << 30) as f64,
        per_token.bytes_per_token(dtype_for_pool),
        pool.blocks * pool.block_size
    );

    let state = AppState {
        engine,
        generation,
        default_max_tokens: args.max_tokens,
    };
    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    println!("  listening on http://{}", listener.local_addr()?);

    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
