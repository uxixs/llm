pub mod backend;
pub mod bench;
pub mod bpe;
pub mod data;
pub mod infer;
pub mod model;
pub mod optim;
pub mod pool;
pub mod rng;
pub mod tui;

pub mod ops;
#[cfg(test)]
mod gradcheck;
pub mod train;
