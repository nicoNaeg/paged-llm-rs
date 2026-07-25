//! Entry point for the inference server.
//!
//! Serving is stage 2 of the build order. What runs today reports which device
//! the build resolved, which is the one thing worth checking before any of it
//! exists: a binary compiled with `--features metal` that silently falls back
//! to the CPU would make every later measurement a lie.

use pagedllm::Backend;

fn main() {
    let backend = Backend::detect();
    println!(
        "pagedllm-server {}, backend {backend}",
        env!("CARGO_PKG_VERSION")
    );
}
