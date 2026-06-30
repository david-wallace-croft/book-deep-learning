use ::tch::nn::{
  self, Adam, LSTM, Linear, LinearConfig, Optimizer, OptimizerConfig, Path,
  RNN, RNNConfig, VarStore,
};
use ::tch::{Device, Kind, TchError, Tensor};

const EPOCH_COUNT: usize = 240;
const HIDDEN_LAYER_SIZE: i64 = 16;
const INPUT_SIZE: i64 = VOCABULARY_LENGTH as i64;
const LEARNING_RATE: f64 = 1e-3;
const OUTPUT_LAYER_SIZE: i64 = VOCABULARY_LENGTH as i64;
const SEED: i64 = 42;
const SEQUENCES_PER_BATCH: usize = 32;
const TIME_STEP_COUNT: usize = 8;
const VOCABULARY_LENGTH: u8 = 6;

fn main() -> Result<(), TchError> {
  let device: Device = Device::cuda_if_available();

  let var_store: VarStore = VarStore::new(device);

  let root_path: &Path<'_> = &var_store.root();

  // let wx: Linear = nn::linear(root / "wx", vocab, hidden, Default::default());

  // let wh: Linear = nn::linear(root / "wh", hidden, hidden, Default::default());

  // let wy: Linear = nn::linear(root / "wy", hidden, vocab, Default::default());

  let rnn_config: RNNConfig = RNNConfig {
    num_layers: 1,
    bidirectional: false,
    batch_first: true,
    ..RNNConfig::default()
  };

  let lstm: LSTM = nn::lstm(
    root_path / "lstm",
    INPUT_SIZE,
    HIDDEN_LAYER_SIZE,
    rnn_config,
  );

  let wy: Linear = nn::linear(
    root_path / "wy",
    HIDDEN_LAYER_SIZE,
    OUTPUT_LAYER_SIZE,
    LinearConfig::default(),
  );

  let mut opt: Optimizer = Adam::default().build(&var_store, LEARNING_RATE)?;

  tch::manual_seed(SEED);

  for epoch in 1..=EPOCH_COUNT {
    let (x_one_hot, y_idx): (Tensor, Tensor) =
      make_batch(SEQUENCES_PER_BATCH, device);

    // let mut h: Tensor = Tensor::zeros(
    //   [
    //     batch, hidden,
    //   ],
    //   (Kind::Float, device),
    // );

    // let mut logits_per_t: Vec<Tensor> = Vec::with_capacity(t_steps as usize);

    // for t in 0..t_steps {
    //   let x_t: Tensor = x_oh.narrow(1, t, 1).squeeze_dim(1);

    //   let a: Tensor = x_t.apply(&wx) + h.apply(&wh);

    //   h = a.tanh();

    //   let logits_t: Tensor = h.apply(&wy);

    //   logits_per_t.push(logits_t);
    // }

    let (h_seq, _state) = lstm.seq(&x_one_hot);

    let logits = h_seq.apply(&wy);

    // let logits: Tensor = Tensor::stack(&logits_per_t, 1);

    let loss: Tensor = logits
      .reshape([
        (SEQUENCES_PER_BATCH * TIME_STEP_COUNT) as i64,
        INPUT_SIZE,
      ])
      .cross_entropy_for_logits(
        &y_idx.reshape([(SEQUENCES_PER_BATCH * TIME_STEP_COUNT) as i64]),
      );

    opt.backward_step(&loss);

    let (x_eval_oh, y_eval_idx): (Tensor, Tensor) = make_batch(1, device);

    // let mut eval_logits_per_t: Vec<Tensor> =
    //   Vec::with_capacity(t_steps as usize);

    if epoch % 10 == 0 {
      // let mut h_eval: Tensor = Tensor::zeros(
      //   [
      //     1, hidden,
      //   ],
      //   (Kind::Float, device),
      // );

      // for t in 0..t_steps {
      //   let x_t: Tensor = x_eval_oh.narrow(1, t, 1).squeeze_dim(1);

      //   h_eval = (x_t.apply(&wx) + h_eval.apply(&wh)).tanh();

      //   eval_logits_per_t.push(h_eval.apply(&wy));
      // }

      // let logits_eval: Tensor = Tensor::stack(&eval_logits_per_t, 1);

      let (h_eval, _st) = lstm.seq(&x_eval_oh);

      let logits_eval = h_eval.apply(&wy);

      let preds = logits_eval.argmax(-1, false);

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

      let accuracy: f64 = correct as f64 / preds_vec.len() as f64;

      let loss_val: f64 = loss.to_device(Device::Cpu).double_value(&[]);

      println!(
        "epoch {:3} | loss {:.4} | eval acc {:>5.1}%",
        epoch,
        loss_val,
        accuracy * 100.
      );
    }
  }

  Ok(())
}

fn make_batch(
  sequences_per_batch: usize,
  device: Device,
) -> (Tensor, Tensor) {
  // Generates random sequences of int64 token indices
  // ranging from zero to (INPUT_LAYER_SIZE minus one) inclusive
  // in a 2D tensor with shape [sequences_per_batch, TIME_STEPS]

  let x_idx: Tensor = Tensor::randint(
    INPUT_SIZE,
    [
      sequences_per_batch as i64,
      TIME_STEP_COUNT as i64,
    ],
    (Kind::Int64, device),
  );

  // Add one and use the modulo operator to wrap within the max vocabulary size

  let y_idx: Tensor = (&x_idx + 1).remainder(INPUT_SIZE);

  // Converts the integer token IDs into a one-hot encoded representation
  // of shape [sequences_per_batch, TIME_STEPS, INPUT_LAYER_SIZE]
  // Example: 2 becomes [0., 0., 1., 0.]

  let x_one_hot: Tensor = x_idx.one_hot(INPUT_SIZE).to_kind(Kind::Float);

  (x_one_hot, y_idx)
}
