use ::tch::nn::{
  self, Adam, Linear, LinearConfig, Optimizer, OptimizerConfig, Path, VarStore,
};
use ::tch::{Device, Kind, TchError, Tensor};

const EPOCH_COUNT: i64 = 120;
const HIDDEN_LAYER_SIZE: i64 = 16;
const INPUT_LAYER_SIZE: i64 = 6; // vocab length
const LEARNING_RATE: f64 = 1e-3;
const SEED: i64 = 42;
const SEQUENCES_PER_BATCH: i64 = 32;
const TIME_STEPS: i64 = 8;

fn main() -> Result<(), TchError> {
  let device: Device = Device::cuda_if_available();

  let var_store: VarStore = VarStore::new(device);

  let root_path: &Path<'_> = &var_store.root();

  let wx: Linear = nn::linear(
    root_path / "wx",
    INPUT_LAYER_SIZE,
    HIDDEN_LAYER_SIZE,
    LinearConfig::default(),
  );

  let wh: Linear = nn::linear(
    root_path / "wh",
    HIDDEN_LAYER_SIZE,
    HIDDEN_LAYER_SIZE,
    LinearConfig::default(),
  );

  let wy: Linear = nn::linear(
    root_path / "wy",
    HIDDEN_LAYER_SIZE,
    INPUT_LAYER_SIZE,
    LinearConfig::default(),
  );

  let mut optimizer: Optimizer =
    Adam::default().build(&var_store, LEARNING_RATE)?;

  tch::manual_seed(SEED);

  for epoch in 1..=EPOCH_COUNT {
    let (_x_idx, y_idx, x_oh): (Tensor, Tensor, Tensor) =
      make_batch(SEQUENCES_PER_BATCH, device);

    let mut h: Tensor = Tensor::zeros(
      [
        SEQUENCES_PER_BATCH,
        HIDDEN_LAYER_SIZE,
      ],
      (Kind::Float, device),
    );

    let mut logits_per_t: Vec<Tensor> = Vec::with_capacity(TIME_STEPS as usize);

    for t in 0..TIME_STEPS {
      let x_t: Tensor = x_oh.narrow(1, t, 1).squeeze_dim(1);

      let a: Tensor = x_t.apply(&wx) + h.apply(&wh);

      h = a.tanh();

      let logits_t: Tensor = h.apply(&wy);

      logits_per_t.push(logits_t);
    }

    let logits: Tensor = Tensor::stack(&logits_per_t, 1);

    let loss: Tensor = logits
      .reshape([
        SEQUENCES_PER_BATCH * TIME_STEPS,
        INPUT_LAYER_SIZE,
      ])
      .cross_entropy_for_logits(
        &y_idx.reshape([SEQUENCES_PER_BATCH * TIME_STEPS]),
      );

    optimizer.backward_step(&loss);

    let (_x_eval_idx, y_eval_idx, x_eval_oh): (Tensor, Tensor, Tensor) =
      make_batch(1, device);

    let mut eval_logits_per_t: Vec<Tensor> =
      Vec::with_capacity(TIME_STEPS as usize);

    if epoch % 10 == 0 {
      let mut h_eval: Tensor = Tensor::zeros(
        [
          1,
          HIDDEN_LAYER_SIZE,
        ],
        (Kind::Float, device),
      );

      for t in 0..TIME_STEPS {
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

// Sets up a sequence prediction task where the goal is to predict the next
// token ID
fn make_batch(
  sequences_per_batch: i64,
  device: Device,
) -> (Tensor, Tensor, Tensor) {
  // Generates random sequences of int64 token indices
  // ranging from zero to INPUT_LAYER_SIZE minus one inclusive
  // in a 2D tensor with shape [sequences_per_batch, TIME_STEPS]
  let x_idx: Tensor = Tensor::randint(
    INPUT_LAYER_SIZE,
    [
      sequences_per_batch,
      TIME_STEPS,
    ],
    (Kind::Int64, device),
  );

  // Add one and use the modulo operator to wrap within the max vocabulary size
  let y_idx: Tensor = (&x_idx + 1).remainder(INPUT_LAYER_SIZE);

  // Converts the integer token IDs into a one-hot encoded representation
  // of shape [sequences_per_batch, TIME_STEPS, INPUT_LAYER_SIZE]
  // Example: 2 becomes [0., 0., 1., 0.]
  let x_onehot: Tensor = x_idx.one_hot(INPUT_LAYER_SIZE).to_kind(Kind::Float);

  (x_idx, y_idx, x_onehot)
}
