#![expect(dead_code)]

use ::tch::{Device, Kind, Tensor};

fn main() {
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
