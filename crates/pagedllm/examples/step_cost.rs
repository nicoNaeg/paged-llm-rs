//! Where a decode step's time goes as the batch grows.
//!
//! A decode step reads every weight of the model once, whatever the batch, so
//! its cost should be nearly flat in the number of rows. When it is not, the
//! thing that scales is worth naming before anything is built on top of it.
//!
//!     cargo run --release --features metal --example step_cost -- models/Qwen3-0.6B

use std::time::Instant;

use pagedllm::{Backend, Batch, CacheConfig, Model, SlotCache};

fn main() -> pagedllm::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: step_cost <model dir>");
        std::process::exit(2)
    });
    let backend = Backend::detect();
    let device = backend.device()?;
    let model = Model::load(&dir, &device)?;
    let config = model.config();
    println!("backend {backend}, {} layers", config.num_hidden_layers);

    let max_seq = 1024;
    let context = 256;
    let prompt: Vec<u32> = (0..context)
        .map(|i| u32::try_from(i % 1000).unwrap() + 10)
        .collect();

    println!(
        "\n{:>5} {:>10} {:>12} {:>14} {:>12} {:>13}",
        "rows", "step ms", "per row ms", "writes only ms", "rest ms", "as prefill ms"
    );
    for rows in [1usize, 2, 4, 8, 16, 32] {
        let mut cache = SlotCache::new(
            CacheConfig {
                slots: rows,
                max_seq,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                layers: config.num_hidden_layers,
            },
            model.dtype(),
            &device,
        )?;
        let slots: Vec<usize> = (0..rows).map(|_| cache.acquire().unwrap()).collect();
        for &slot in &slots {
            model.forward_batch(&Batch::prefill(prompt.clone(), slot, 0), &cache)?;
            cache.advance(&[slot], context);
        }

        let decode = |cache: &SlotCache, starts: Vec<usize>| -> pagedllm::Result<()> {
            let batch = Batch::decode(vec![7; rows], slots.clone(), starts);
            model.forward_batch(&batch, cache)?;
            Ok(())
        };

        // Warm, then timed. The first dispatch of a shape pays for pipeline
        // setup that no later one repeats.
        let starts: Vec<usize> = slots.iter().map(|&s| cache.length(s)).collect();
        decode(&cache, starts.clone())?;

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            decode(&cache, starts.clone())?;
        }
        let step_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iterations);

        // The same batch with only the cache writes, which is what scales with
        // the rows if anything does: one call per row, per layer, twice.
        let start = Instant::now();
        for _ in 0..iterations {
            cache.write_probe(&slots, &starts)?;
        }
        let writes_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iterations);

        // The same number of rows through the model, but as one sequence's
        // prompt instead of one token each. Identical matrix work; a different
        // shape of attention and one row of cache instead of many.
        let mut fresh = SlotCache::new(
            CacheConfig {
                slots: 1,
                max_seq,
                kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                layers: config.num_hidden_layers,
            },
            model.dtype(),
            &device,
        )?;
        let one = fresh.acquire().unwrap();
        model.forward_batch(&Batch::prefill(prompt.clone(), one, 0), &fresh)?;
        fresh.advance(&[one], context);
        let chunk: Vec<u32> = vec![7; rows];
        let prefill = Batch::prefill(chunk, one, context);
        model.forward_batch(&prefill, &fresh)?;
        let start = Instant::now();
        for _ in 0..iterations {
            model.forward_batch(&prefill, &fresh)?;
        }
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iterations);

        println!(
            "{rows:>5} {step_ms:>10.2} {:>12.2} {writes_ms:>14.2} {:>12.2} {prefill_ms:>13.2}",
            step_ms / rows as f64,
            step_ms - writes_ms
        );
    }
    Ok(())
}
