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

  let (_x_idx, y_idx, x_oh) = make_batch(batch, t_steps, vocab, device);

  let mut h = Tensor::zeros(
    [
      batch, hidden,
    ],
    (Kind::Float, device),
  );

  let mut logits_per_t: Vec<Tensor> = Vec::with_capacity(t_steps as usize);

  for t in 0..t_steps {
    let x_t = x_oh.narrow(1, t, 1).squeeze_dim(1);

    let a = x_t.apply(&wx) + h.apply(&wh);

    h = a.tanh();

    let logits_t = h.apply(&wy);

    logits_per_t.push(logits_t);
  }

  let logits = Tensor::stack(&logits_per_t, 1);

  let loss = logits
    .reshape([
      batch * t_steps,
      vocab,
    ])
    .cross_entropy_for_logits(&y_idx.reshape([batch * t_steps]));

  opt.backward_step(&loss);

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
