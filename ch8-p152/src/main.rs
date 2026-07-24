use self::sum_mod_transformer::SumModTransformer;
use ::tch::nn::{Adam, Optimizer, OptimizerConfig, Path, VarStore};
use ::tch::{Device, Kind, Result, Tensor};

mod encoder_block;
mod multi_head_self_attention;
mod sum_mod_transformer;

// Accuracy does not significantly improve at these settings:
// EPOCHS = 10_000, FF_DIMENSIONS = 1_024, MODEL_DIMENSIONS = 256

const BATCH_SIZE: i64 = 128;
const CLASS_COUNT: i64 = 5;
const DROPOUT_PROBABILITY: f64 = 0.1;
const EPOCHS: i64 = 300;
const FF_DIMENSIONS: i64 = 256;
const HEADS: i64 = 4;
const LAYERS: i64 = 2;
const LEARNING_RATE: f64 = 1e-3;
const MODEL_DIMENSIONS: i64 = 64;
const RANDOM_SEED: i64 = 42;
const TIME_STEPS: i64 = 16;
const VOCABULARY_SIZE: i64 = 10;

fn main() -> Result<()> {
  tch::manual_seed(RANDOM_SEED);

  let device: Device = Device::cuda_if_available();

  // println!("device: {device:?}");

  let var_store: VarStore = VarStore::new(device);

  let root_path: &Path<'_> = &var_store.root();

  let model: SumModTransformer = SumModTransformer::new(
    root_path,
    VOCABULARY_SIZE,
    MODEL_DIMENSIONS,
    HEADS,
    FF_DIMENSIONS,
    LAYERS,
    CLASS_COUNT,
    TIME_STEPS,
    DROPOUT_PROBABILITY,
    device,
  );

  let mut opt: Optimizer =
    Adam::default().build(&var_store, LEARNING_RATE).unwrap();

  for epoch in 1..=EPOCHS {
    // Returns a tensor filled with random integers generated uniformly between
    // low (inclusive) and high (exclusive).
    // https://docs.pytorch.org/docs/main/generated/torch.randint.html
    let x_idx: Tensor = Tensor::randint(
      VOCABULARY_SIZE,
      [
        BATCH_SIZE, TIME_STEPS,
      ],
      (Kind::Int64, device),
    );

    // x_idx is [[128, 16], Int64] with random values between 0 and 9 inclusive
    // println!("x_idx: {x_idx}");

    let y: Tensor = x_idx
      .to_kind(Kind::Float)
      .sum_dim_intlist([1].as_slice(), false, Kind::Float)
      .remainder(CLASS_COUNT as f64)
      .to_kind(Kind::Int64);

    let logits: Tensor = model.forward_t(&x_idx, true);

    let loss: Tensor = logits.cross_entropy_for_logits(&y);

    opt.backward_step(&loss);

    if epoch % 10 == 0 || epoch == 1 {
      let acc: f64 = accuracy_from_logits(&logits, &y);

      let loss_cpu_tensor: Tensor = loss.to_device(Device::Cpu);

      let l: f64 = loss_cpu_tensor.double_value(&[]);

      println!(
        "epoch {:4} | loss {:6.4} | acc {:5.1}%",
        epoch,
        l,
        acc * 100.
      );
    }
  }

  let test_b: i64 = 8;

  let x_idx: Tensor = Tensor::randint(
    VOCABULARY_SIZE,
    [
      test_b, TIME_STEPS,
    ],
    (Kind::Int64, device),
  );

  let y: Tensor = x_idx
    .to_kind(Kind::Float)
    .sum_dim_intlist([1].as_slice(), false, Kind::Float)
    .remainder(CLASS_COUNT as f64)
    .to_kind(Kind::Int64);

  let logits: Tensor = model.forward_t(&x_idx, false);

  let pred: Tensor = logits.argmax(-1, false);

  let y_cpu_tensor: Tensor = y.to_device(Device::Cpu);

  let true_labels: Vec<i64> = Vec::<i64>::try_from(y_cpu_tensor).unwrap();

  println!("true labels: {true_labels:?}");

  let pred_cpu_tensor: Tensor = pred.to_device(Device::Cpu);

  let pred_labels: Vec<i64> = Vec::<i64>::try_from(pred_cpu_tensor).unwrap();

  println!("pred labels: {pred_labels:?}");

  Ok(())
}

fn accuracy_from_logits(
  logits: &Tensor,
  y: &Tensor,
) -> f64 {
  // Logits is a 128 by 5 matrix of positive and negative float values
  // println!("logits: {logits}");

  // y is a vector of 128 integer values
  // println!("y: {y}");

  // Returns the indices of the max value of all elements in the input tensor
  // https://docs.pytorch.org/docs/main/generated/torch.argmax.html
  // The first argument is the dimension to reduce
  let pred: Tensor = logits.argmax(-1, false);

  // pred is a vector of 128 integer values
  // println!("pred: {pred}");

  let correct_bool: Tensor = pred.eq_tensor(y);

  // correct_bool is a vector of 128 boolean values
  // println!("correct_bool: {correct_bool}");

  let correct_float: Tensor = correct_bool.to_kind(Kind::Float);

  // correct float is a vector 128 float values, each either zero or one
  // println!("correct_float: {correct_float}");

  let correct: Tensor = correct_float.mean(Kind::Float);

  // correct is a scalar
  // println!("correct: {correct}");

  // Returns a double value on tensors holding a single element
  correct.double_value(&[])
}
