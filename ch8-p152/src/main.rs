#![expect(dead_code)]
#![expect(unused_imports)]
#![expect(unused_mut)]
#![expect(unused_variables)]

use self::sum_mod_transformer::SumModTransformer;
use ::tch::nn::{
  self, Adam, ModuleT, Optimizer, OptimizerConfig, Path, Sequential, VarStore,
};
use ::tch::{Device, Kind, Result, TchError, Tensor};

mod encoder_block;
mod mhsa;
mod sum_mod_transformer;

const BATCH: i64 = 128;
const D_FF: i64 = 256;
const D_MODEL: i64 = 64;
const DROPOUT_P: f64 = 0.1;
const EPOCHS: i64 = 300;
const LEARNING_RATE: f64 = 1e-3;
const N_CLASSES: i64 = 5;
const N_HEADS: i64 = 4;
const N_LAYERS: i64 = 2;
const SEED: i64 = 42;
const T_STEPS: i64 = 16;
const VOCAB: i64 = 10;

fn main() -> Result<()> {
  tch::manual_seed(SEED);

  let device: Device = Device::cuda_if_available();

  let mut var_store: VarStore = VarStore::new(device);

  let root: &Path<'_> = &var_store.root();

  let model = SumModTransformer::new(
    root, VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, N_CLASSES, T_STEPS,
    DROPOUT_P, device,
  );

  todo!()
}
