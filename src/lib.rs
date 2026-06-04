//! bebelm — CPU-only, pure-Rust inference for Liquid AI LFM2.5-8B-A1B (Q4_K_M).
//!
//! Library surface: the GGUF loader, dtype/block sizing, and the compute kernels. The
//! `bebelm` binary (`src/main.rs`) is a thin CLI over this crate.

pub mod cache;
pub mod config;
pub mod gguf;
pub mod kernels;
pub mod model;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;
