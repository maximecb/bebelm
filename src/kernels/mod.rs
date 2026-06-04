//! Compute kernels. Single-core scalar `f32` first; SIMD/threads come later (design.md).

pub mod activation;
pub mod dequant;
pub mod elementwise;
pub mod matmul;
pub mod rmsnorm;
pub mod softmax;
