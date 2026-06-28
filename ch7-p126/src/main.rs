use ::tch::nn::{
  self, Adam, Linear, LinearConfig, Optimizer, OptimizerConfig, Path, VarStore,
};
use ::tch::{Device, Kind, TchError, Tensor};

const EPOCH_COUNT: usize = 240;
const HIDDEN_LAYER_SIZE: i64 = 16;
const INPUT_SIZE: i64 = VOCABULARY_LENGTH as i64;
const INPUT_LAYER_SIZE: i64 = 16;
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

  let wx: Linear = nn::linear(
    root_path / "wx",
    INPUT_SIZE,
    INPUT_LAYER_SIZE,
    LinearConfig::default(),
  );

  let wh: Linear = nn::linear(
    root_path / "wh",
    INPUT_LAYER_SIZE,
    HIDDEN_LAYER_SIZE,
    LinearConfig::default(),
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

    // The hidden state h is 32 by 16 and starts off as all zeroes

    let mut hidden_state: Tensor = Tensor::zeros(
      [
        SEQUENCES_PER_BATCH as i64,
        HIDDEN_LAYER_SIZE,
      ],
      (Kind::Float, device),
    );

    let mut logits_per_time_step: Vec<Tensor> =
      Vec::with_capacity(TIME_STEP_COUNT);

    for time_step_index in 0..TIME_STEP_COUNT {
      // It extracts a slice of the input tensor (x_one_hot) at the current
      // time step using .narrow(),
      // and removes the extra dimension using .squeeze_dim()
      // x_t is a 32 by 6

      let x_t: Tensor = x_one_hot
        .narrow(1, time_step_index as i64, 1)
        .squeeze_dim(1);

      // println!("{x_t}");

      // x_t is a 32 by 6
      // wx is a 6 by 16
      // So x_t times wx is a 32 by 16

      let a1: Tensor = x_t.apply(&wx);

      // println!("{a1}");

      // h is a 32 by 16
      // wh is a 16 by 16
      // So h times wh is a 32 by 16

      let a2: Tensor = hidden_state.apply(&wh);

      // a is a 32 by 16

      let a: Tensor = a1 + a2;

      // It computes the next hidden state

      hidden_state = a.tanh();

      // println!("{h}");

      // It passes the new hidden state through an output weight matrix wy to
      // get the raw unnormalized predictions (logits) for this time step,
      // then pushes them to the vector

      let logits_at_time_step_t: Tensor = hidden_state.apply(&wy);

      logits_per_time_step.push(logits_at_time_step_t);
    }

    // logits_per_time_step is a Vec of length 8 containing 32 by 6 Tensors

    // logits is a 32 by 8 by 6 Tensor

    let logits: Tensor = Tensor::stack(&logits_per_time_step, 1);

    // println!("{logits}");

    // logits_reshaped is (32 times 8 = 256) by 6

    let logits_reshaped: Tensor = logits.reshape([
      (SEQUENCES_PER_BATCH * TIME_STEP_COUNT) as i64,
      INPUT_SIZE,
    ]);

    // println!("{logits_reshaped}");

    // y_idx is 32 by 8

    // println!("{y_idx}");

    // y_idx_reshaped is (32 * 8 = 256) by 1

    let y_idx_reshaped: Tensor =
      y_idx.reshape([(SEQUENCES_PER_BATCH * TIME_STEP_COUNT) as i64]);

    // println!("{y_idx_reshaped}");

    // log_softmax is 256 by 6

    // let log_softmax = logits_reshaped.log_softmax(-1, Kind::Float);

    // println!("{log_softmax}");

    // loss is 1 by 1

    let loss: Tensor =
      logits_reshaped.cross_entropy_for_logits(&y_idx_reshaped);

    // println!("{loss}");

    optimizer.backward_step(&loss);

    if epoch % 10 == 0 {
      evaluate(device, epoch, &loss, &wh, &wx, &wy)?;
    }
  }

  Ok(())
}

fn evaluate(
  device: Device,
  epoch: usize,
  loss: &Tensor,
  wh: &Linear,
  wx: &Linear,
  wy: &Linear,
) -> Result<(), TchError> {
  let (x_eval_one_hot, y_eval_idx): (Tensor, Tensor) = make_batch(1, device);

  // println!("{x_eval_idx}");

  // println!("{y_eval_idx}");

  // println!("{x_eval_one_hot}");

  // Pre-allocates a vector to store the output predictions (logits) for each
  // individual time step

  let mut eval_logits_per_time_step: Vec<Tensor> =
    Vec::with_capacity(TIME_STEP_COUNT);

  // Initializes the hidden state to a tensor of zeros.
  // This represents the network's "memory" before it sees any data

  let mut h_eval: Tensor = Tensor::zeros(
    [
      1,
      HIDDEN_LAYER_SIZE,
    ],
    (Kind::Float, device),
  );

  for time_step_index in 0..TIME_STEP_COUNT {
    // It extracts a slice of the input tensor (x_eval_one_hot) at the current
    // time step using .narrow(),
    // and removes the extra dimension using .squeeze_dim()

    let x_t: Tensor = x_eval_one_hot
      .narrow(1, time_step_index as i64, 1)
      .squeeze_dim(1);

    // println!("{x_t}");

    // It computes the next hidden state

    h_eval = (x_t.apply(wx) + h_eval.apply(wh)).tanh();

    // It passes the new hidden state through an output weight matrix wy to get
    // the raw unnormalized predictions (logits) for this time step,
    // then pushes them to the vector

    eval_logits_per_time_step.push(h_eval.apply(wy));
  }

  // Combines the list of individual time-step logit tensors back into a single
  // unified tensor along the time dimension

  let logits_eval: Tensor = Tensor::stack(&eval_logits_per_time_step, 1);

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
