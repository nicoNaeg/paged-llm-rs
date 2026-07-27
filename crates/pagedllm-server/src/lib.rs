//! An `OpenAI`-compatible HTTP server for the `pagedllm` engine.
//!
//! The protocol is the point rather than a convenience. Speaking the API every
//! client already drives means the comparison against another engine on this
//! machine needs no harness written here, and anyone can reproduce it with the
//! tool they already point at other servers.

pub mod engine;
pub mod openai;
pub mod routes;

pub use engine::{Engine, Event};
pub use routes::{AppState, router};
