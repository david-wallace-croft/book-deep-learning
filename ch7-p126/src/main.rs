#![expect(dead_code)]
#![expect(unused_mut)]
#![expect(unused_variables)]

use ::tch::nn::{
  self, Adam, Linear, Optimizer, OptimizerConfig, Path, VarStore,
};
use ::tch::{Device, Kind, TchError, Tensor};

fn main() -> Result<(), TchError> {
  let device: Device = Device::cuda_if_available();

  let vs: VarStore = VarStore::new(device);

  let root: &Path<'_> = &vs.root();

  let vocab: i64 = 6;

  let hidden: i64 = 16;

  let t_steps: i64 = 8;

  let batch: i64 = 32;

  let epochs: i64 = 120;

  let wx: Linear = nn::linear(root / "wx", vocab, hidden, Default::default());

  let wh: Linear = nn::linear(root / "wh", hidden, hidden, Default::default());

  let wy: Linear = nn::linear(root / "wy", hidden, vocab, Default::default());

  let mut opt: Optimizer = Adam::default().build(&vs, 1e-3)?;

  todo!()
}

fn make_batch(
  batch: i64,
  t_steps: i64,
  vocab: i64,
  device: Device,
) -> (Tensor, Tensor, Tensor) {
  let x_idx = Tensor::randint(
    vocab,
    [
      batch, t_steps,
    ],
    (Kind::Int64, device),
  );

  let y_idx = (&x_idx + 1).remainder(vocab);

  let x_onehot = x_idx.one_hot(vocab).to_kind(Kind::Float);

  (x_idx, y_idx, x_onehot)
}
