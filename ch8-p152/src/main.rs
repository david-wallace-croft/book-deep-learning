#![expect(dead_code)]
#![expect(unused_imports)]
#![expect(unused_variables)]

use ::tch::nn::{
  self, Adam, ModuleT, Optimizer, OptimizerConfig, Path, Sequential, VarStore,
};
use ::tch::{Device, Kind, Result, TchError, Tensor};

const BATCH: usize = 128;
const D_FF: usize = 256;
const D_MODEL: usize = 64;
const DROPOUT_P: f64 = 0.1;
const EPOCHS: usize = 300;
const LEARNING_RATE: f64 = 1e-3;
const N_CLASSES: usize = 5;
const N_HEADS: usize = 4;
const N_LAYERS: usize = 2;
const SEED: i64 = 42;
const T_STEPS: usize = 16;
const VOCAB: usize = 10;

fn main() -> Result<()> {
  tch::manual_seed(SEED);

  let device: Device = Device::cuda_if_available();

  todo!()
}
