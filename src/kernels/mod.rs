//! Compute kernels. Single-core scalar `f32` first; SIMD/threads come later (design.md).

pub mod dequant;
pub mod matmul;
pub mod rmsnorm;
