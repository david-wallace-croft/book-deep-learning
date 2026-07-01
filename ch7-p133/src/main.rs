use ::tch::nn::{
  self, Adam, LSTM, LSTMState, Linear, LinearConfig, Optimizer,
  OptimizerConfig, Path, RNN, RNNConfig, VarStore,
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

  let mut optimizer: Optimizer =
    Adam::default().build(&var_store, LEARNING_RATE)?;

  tch::manual_seed(SEED);

  for epoch in 1..=EPOCH_COUNT {
    let (x_one_hot, y_idx): (Tensor, Tensor) =
      make_batch(SEQUENCES_PER_BATCH, device);

    let (h_seq, _state): (Tensor, LSTMState) = lstm.seq(&x_one_hot);

    let logits: Tensor = h_seq.apply(&wy);

    let logits_reshaped: Tensor = logits.reshape([
      (SEQUENCES_PER_BATCH * TIME_STEP_COUNT) as i64,
      INPUT_SIZE,
    ]);

    let y_idx_reshaped: Tensor =
      y_idx.reshape([(SEQUENCES_PER_BATCH * TIME_STEP_COUNT) as i64]);

    let loss: Tensor =
      logits_reshaped.cross_entropy_for_logits(&y_idx_reshaped);

    optimizer.backward_step(&loss);

    if epoch % 10 == 0 {
      evaluate(device, epoch, &loss, &lstm, &wy)?;
    }
  }

  Ok(())
}

fn evaluate(
  device: Device,
  epoch: usize,
  loss: &Tensor,
  lstm: &LSTM,
  wy: &Linear,
) -> Result<(), TchError> {
  let (x_eval_one_hot, y_eval_idx): (Tensor, Tensor) = make_batch(1, device);

  let (h_eval, _st): (Tensor, LSTMState) = lstm.seq(&x_eval_one_hot);

  let logits_eval: Tensor = h_eval.apply(wy);

  // Finds the index of the highest value along the last dimension (-1).
  // This converts the model's raw scores into actual class/character
  // predictions (e.g., if predicting text, it picks the most likely next
  // character).

  let preds: Tensor = logits_eval.argmax(-1, false);

  // Moves the prediction tensor from the GPU/device back to the CPU,
  // flattens it into a 1D shape (.view([-1])),
  // and converts it into a standard Rust Vec<i64> so you can easily print or
  // use the results in the rest of your Rust application

  let preds_vec: Vec<i64> = preds
    .to_device(Device::Cpu)
    .view([-1])
    .iter::<i64>()?
    .collect();

  // The y_eval_idx tensor contains the actual, true labels
  // (the ground truth targets).
  // Just like it did for the predictions (preds_vec), this code moves the
  // target tensor to the CPU,
  // flattens it into a 1D layout (.view([-1])),
  // and collects it into a standard Rust vector (Vec<i64>)

  let y_vec: Vec<i64> = y_eval_idx
    .to_device(Device::Cpu)
    .view([-1])
    .iter::<i64>()?
    .collect();

  // It pairs up each predicted value from preds_vec with its corresponding true
  // value from y_vec.
  // It keeps only the pairs where the prediction matches the true label.
  // Counts the total number of correct predictions.

  let correct: usize = preds_vec
    .iter()
    .zip(y_vec.iter())
    .filter(|(a, b)| a == b)
    .count();

  // Divides the number of correct predictions by the total number of items
  // to get the accuracy percentage as a decimal (e.g., 0.85 for 85%)

  let accuracy: f64 = correct as f64 / preds_vec.len() as f64;

  // Takes the loss tensor (which represents how poorly/well the model performed
  // during this pass), moves it to the CPU, and extracts it as a standard Rust
  // f64 scalar value.

  let loss_val: f64 = loss.to_device(Device::Cpu).double_value(&[]);

  println!(
    "epoch {:3} | loss {:.4} | eval accuracy {:>5.1}%",
    epoch,
    loss_val,
    accuracy * 100.
  );

  Ok(())
}

// Sets up a sequence prediction task where the goal is to predict the next
// token ID
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
