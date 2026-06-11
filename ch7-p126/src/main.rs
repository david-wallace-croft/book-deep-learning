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

  tch::manual_seed(42);

  for epoch in 1..=epochs {
    let (_x_idx, y_idx, x_oh): (Tensor, Tensor, Tensor) =
      make_batch(batch, t_steps, vocab, device);

    let mut h: Tensor = Tensor::zeros(
      [
        batch, hidden,
      ],
      (Kind::Float, device),
    );

    let mut logits_per_t: Vec<Tensor> = Vec::with_capacity(t_steps as usize);

    for t in 0..t_steps {
      let x_t: Tensor = x_oh.narrow(1, t, 1).squeeze_dim(1);

      let a: Tensor = x_t.apply(&wx) + h.apply(&wh);

      h = a.tanh();

      let logits_t: Tensor = h.apply(&wy);

      logits_per_t.push(logits_t);
    }

    let logits: Tensor = Tensor::stack(&logits_per_t, 1);

    let loss: Tensor = logits
      .reshape([
        batch * t_steps,
        vocab,
      ])
      .cross_entropy_for_logits(&y_idx.reshape([batch * t_steps]));

    opt.backward_step(&loss);

    let (_x_eval_idx, y_eval_idx, x_eval_oh): (Tensor, Tensor, Tensor) =
      make_batch(1, t_steps, vocab, device);

    let mut eval_logits_per_t: Vec<Tensor> =
      Vec::with_capacity(t_steps as usize);

    if epoch % 10 == 0 {
      let mut h_eval: Tensor = Tensor::zeros(
        [
          1, hidden,
        ],
        (Kind::Float, device),
      );

      for t in 0..t_steps {
        let x_t: Tensor = x_eval_oh.narrow(1, t, 1).squeeze_dim(1);

        h_eval = (x_t.apply(&wx) + h_eval.apply(&wh)).tanh();

        eval_logits_per_t.push(h_eval.apply(&wy));
      }

      let logits_eval: Tensor = Tensor::stack(&eval_logits_per_t, 1);

      let preds: Tensor = logits_eval.argmax(-1, false);

      let preds_vec: Vec<i64> = preds
        .to_device(Device::Cpu)
        .view([-1])
        .iter::<i64>()?
        .collect();

      let y_vec: Vec<i64> = y_eval_idx
        .to_device(Device::Cpu)
        .view([-1])
        .iter::<i64>()?
        .collect();

      let correct: usize = preds_vec
        .iter()
        .zip(y_vec.iter())
        .filter(|(a, b)| a == b)
        .count();

      let acc: f64 = correct as f64 / preds_vec.len() as f64;

      let loss_val: f64 = loss.to_device(Device::Cpu).double_value(&[]);

      println!(
        "epoch {:3} | loss {:.4} | eval acc {:>5.1}%",
        epoch,
        loss_val,
        acc * 100.
      );
    }
  }

  Ok(())
}

fn make_batch(
  batch: i64,
  t_steps: i64,
  vocab: i64,
  device: Device,
) -> (Tensor, Tensor, Tensor) {
  let x_idx: Tensor = Tensor::randint(
    vocab,
    [
      batch, t_steps,
    ],
    (Kind::Int64, device),
  );

  let y_idx: Tensor = (&x_idx + 1).remainder(vocab);

  let x_onehot: Tensor = x_idx.one_hot(vocab).to_kind(Kind::Float);

  (x_idx, y_idx, x_onehot)
}
