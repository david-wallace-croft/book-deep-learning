#![expect(dead_code)]
#![expect(unused_imports)]
#![expect(unused_variables)]

use self::example::Example;
use self::vocab::Vocab;
use ::tch::nn::{Adam, Optimizer, OptimizerConfig, Path, VarStore};
use ::tch::{Device, Kind, Result, Tensor};

mod example;
mod vocab;

const BATCH_SIZE: i64 = 4;
const D_FF: i64 = 128;
const D_MODEL: i64 = 64;
const EPOCHS: i64 = 300;
const LEARNING_RATE: f64 = 1e-3;
const N_HEADS: i64 = 4;
const SEQ_LEN: usize = 8;
const RANDOM_SEED: i64 = 42;

fn main() -> Result<()> {
  tch::manual_seed(RANDOM_SEED);

  let device: Device = Device::Cpu;

  todo!()
}

fn toy_data(seq_len: usize) -> (Vocab, Vec<Example>) {
  todo!()
}
