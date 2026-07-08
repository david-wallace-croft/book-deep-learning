#![expect(dead_code)]
#![expect(unused_imports)]
#![expect(unused_mut)]
#![expect(unused_variables)]
use ::tch::nn::{
  self, Adam, LSTM, LSTMState, Linear, LinearConfig, Optimizer,
  OptimizerConfig, Path, RNN, RNNConfig, VarStore,
};
use ::tch::{Device, Kind, TchError, Tensor};

const BATCH: i64 = 128;
const ITERS: i64 = 2_000;
const LEARNING_RATE: f64 = 2e-4;
const PRINT_EVERY: i64 = 100;
const SEED: i64 = 42;
const Z_DIM: i64 = 8;

fn main() -> Result<(), TchError> {
  tch::manual_seed(SEED);

  let device: Device = Device::cuda_if_available();

  let var_store_discriminator: VarStore = VarStore::new(device);

  let var_store_generator: VarStore = VarStore::new(device);

  let mut optimizer_discriminator: Optimizer =
    Adam::default().build(&var_store_discriminator, LEARNING_RATE)?;

  let mut optimizer_generator: Optimizer =
    Adam::default().build(&var_store_generator, LEARNING_RATE)?;

  for step in 1..=ITERS {
    let x_real: Tensor = sample_real(BATCH, device)?;

    todo!()
  }

  todo!()
}

fn sample_real(
  batch: i64,
  device: Device,
) -> Result<Tensor, TchError> {
  let half: i64 = batch / 2;

  let std: f64 = 0.5_f64;

  let mean1 = Tensor::f_from_slice(&[
    -2.0_f32, 0.,
  ])?
  .to_device(device)
  .view([
    1, 2,
  ]);

  let mean2 = Tensor::f_from_slice(&[
    2.0_f32, 0.,
  ])?
  .to_device(device)
  .view([
    1, 2,
  ]);

  let x1 = Tensor::randn(
    [
      half, 2,
    ],
    (Kind::Float, device),
  ) * std
    + &mean1;

  let x2 = Tensor::randn(
    [
      batch - half,
      2,
    ],
    (Kind::Float, device),
  ) * std
    + &mean2;

  Ok(Tensor::cat(
    &[
      x1, x2,
    ],
    0,
  ))
}
