//! Where a decode step's time goes as the batch grows, and what paging adds.
//!
//! A decode step reads every weight of the model once, whatever the batch, so
//! its cost should be nearly flat in the number of rows. Stage 3 measured that
//! it was not and did not find out why, and this is where that question came
//! from. It has since been answered and the answer was the gather: the tensor
//! path copies every resident sequence's keys and values into one rectangle per
//! layer per step, which is proportional to the batch and cancels what batching
//! buys. The comparison stays because it is what shows the kernel removing it.
//!
//! It runs at two block sizes. A block as wide as the context is one block a
//! sequence, which is the reservation stage 3 measured; a block of sixteen is
//! paging. Their difference is what paging costs on the read path, with
//! everything else held still.
//!
//!     cargo run --release --features metal --example step_cost -- models/Qwen3-0.6B

use std::time::Instant;

use pagedllm::{
    AttentionKind, Backend, Batch, BlockAllocator, BlockTable, CacheConfig, Model, PagedCache,
};

const CONTEXT: usize = 256;
const ITERATIONS: u32 = 10;

fn main() -> pagedllm::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: step_cost <model dir>");
        std::process::exit(2)
    });
    let backend = Backend::detect();
    let device = backend.device()?;
    let mut model = Model::load(&dir, &device)?;
    let (layers, kv_heads, head_dim) = {
        let config = model.config();
        (
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
        )
    };
    println!("backend {backend}, {layers} layers");

    let prompt: Vec<u32> = (0..CONTEXT)
        .map(|i| u32::try_from(i % 1000).unwrap() + 10)
        .collect();

    for (block_size, kind) in [
        (1024usize, AttentionKind::Tensor),
        (16, AttentionKind::Tensor),
        (16, AttentionKind::Kernel),
    ] {
        model.set_attention(kind);
        println!(
            "\n{}",
            match (block_size, kind) {
                (1024, _) => "a reservation, gathered by the tensor path",
                (_, AttentionKind::Tensor) => "paging, gathered by the tensor path",
                (_, AttentionKind::Kernel) => "paging, read in place by the kernel",
            }
        );
        println!(
            "{:>5} {:>10} {:>12} {:>15}",
            "rows", "step ms", "per row ms", "as prefill ms"
        );

        for rows in [1usize, 2, 4, 8, 16, 32] {
            let per_row = (CONTEXT + 64).div_ceil(block_size);
            let blocks = per_row * (rows + 1);
            let cache = PagedCache::new(
                CacheConfig {
                    block_size,
                    blocks,
                    kv_heads,
                    head_dim,
                    layers,
                },
                model.dtype(),
                &device,
            )?;
            let mut allocator = BlockAllocator::new(blocks);

            let mut tables = Vec::with_capacity(rows);
            for _ in 0..rows {
                let mut table = BlockTable::new(block_size);
                // Room for the decode token and for the prefill chunk timed below.
                for _ in 0..table.blocks_needed(CONTEXT + 64) {
                    table.push(allocator.allocate().expect("sized above"));
                }
                let batch = Batch::new(prompt.clone(), CONTEXT, &[&table], block_size)?;
                model.forward_batch(&batch, &cache)?;
                table.advance(CONTEXT)?;
                tables.push(table);
            }

            let refs: Vec<&BlockTable> = tables.iter().collect();
            let decode = Batch::new(vec![7; rows], 1, &refs, block_size)?;
            model.forward_batch(&decode, &cache)?;
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                model.forward_batch(&decode, &cache)?;
            }
            let step_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(ITERATIONS);

            // The same rows through the model as one sequence's prompt rather
            // than one token each. Identical matrix work, a different shape of
            // attention, and one row of cache instead of many.
            let chunk = Batch::new(vec![7; rows], rows, &refs[..1], block_size)?;
            model.forward_batch(&chunk, &cache)?;
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                model.forward_batch(&chunk, &cache)?;
            }
            let prefill_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(ITERATIONS);

            println!(
                "{rows:>5} {step_ms:>10.2} {:>12.2} {prefill_ms:>15.2}",
                step_ms / rows as f64
            );
        }
    }
    Ok(())
}
