#![expect(unused_imports)]
#![expect(unused_variables)]

use ::tch::nn::{Adam, Optimizer, OptimizerConfig, Path, VarStore};
use ::tch::{Device, Kind, Result, Tensor};

const RANDOM_SEED: i64 = 42;

fn main() -> Result<()> {
  tch::manual_seed(RANDOM_SEED);

  let device: Device = Device::Cpu;

  todo!()
}
