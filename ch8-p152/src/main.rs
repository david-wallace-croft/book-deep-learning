use self::sum_modulo_transformer::SumModuloTransformer;
use ::tch::nn::{Adam, Optimizer, OptimizerConfig, Path, VarStore};
use ::tch::{Device, Kind, Result, Tensor};

mod encoder_block;
mod multi_head_self_attention;
mod sum_modulo_transformer;

// Accuracy does not significantly improve at these settings:
// EPOCHS = 10_000, FEED_FORWARD_DIMENSIONS = 1_024, MODEL_DIMENSIONS = 256

const BATCH_SIZE_TEST: i64 = 8;
const BATCH_SIZE_TRAIN: i64 = 128;
const CLASS_COUNT: i64 = 5;
const DROPOUT_PROBABILITY: f64 = 0.1;
const EPOCHS: usize = 30;
const FEED_FORWARD_DIMENSIONS: i64 = 256;
const HEADS: i64 = 4;
const LAYERS: i64 = 2;
const LEARNING_RATE: f64 = 1e-3;
const MODEL_DIMENSIONS: i64 = 64;
const PRINT_INTERVAL: usize = 10;
const RANDOM_SEED: i64 = 42;
const TIME_STEPS: i64 = 16;
const VOCABULARY_SIZE: i64 = 10;

fn main() -> Result<()> {
  tch::manual_seed(RANDOM_SEED);

  let device: Device = Device::cuda_if_available();

  // println!("device: {device:?}");

  let var_store: VarStore = VarStore::new(device);

  let root_path: &Path<'_> = &var_store.root();

  let model: SumModuloTransformer = SumModuloTransformer::new(
    root_path,
    VOCABULARY_SIZE,
    MODEL_DIMENSIONS,
    HEADS,
    FEED_FORWARD_DIMENSIONS,
    LAYERS,
    CLASS_COUNT,
    TIME_STEPS,
    DROPOUT_PROBABILITY,
    device,
  );

  let mut optimizer: Optimizer =
    Adam::default().build(&var_store, LEARNING_RATE).unwrap();

  for epoch in 1..=EPOCHS {
    // Returns a tensor filled with random integers generated uniformly between
    // low (inclusive) and high (exclusive).
    // https://docs.pytorch.org/docs/main/generated/torch.randint.html
    let x_idx: Tensor = Tensor::randint(
      VOCABULARY_SIZE,
      [
        BATCH_SIZE_TRAIN,
        TIME_STEPS,
      ],
      (Kind::Int64, device),
    );

    let y: Tensor = calculate_true_classes(&x_idx);

    let logits: Tensor = model.forward_t(&x_idx, true);

    // logits is [[128, 5], Float] where the values are positive and negative
    // println!("logits: {logits}");

    let loss_tensor: Tensor = logits.cross_entropy_for_logits(&y);

    // loss_tensor is a scalar positive number
    // println!("loss: {loss}");

    optimizer.backward_step(&loss_tensor);

    if epoch % PRINT_INTERVAL == 0 {
      let accuracy: f64 = 100. * accuracy_from_logits(&logits, &y);

      let loss_cpu_tensor: Tensor = loss_tensor.to_device(Device::Cpu);

      let loss: f64 = loss_cpu_tensor.double_value(&[]);

      println!("epoch {epoch} | loss {loss:.3} | accuracy {accuracy:.1}%");
    }
  }

  let x_idx: Tensor = Tensor::randint(
    VOCABULARY_SIZE,
    [
      BATCH_SIZE_TEST,
      TIME_STEPS,
    ],
    (Kind::Int64, device),
  );

  let y: Tensor = calculate_true_classes(&x_idx);

  let true_labels: Vec<i64> = to_vec_i64(y);

  println!("true labels: {true_labels:?}");

  let logits: Tensor = model.forward_t(&x_idx, false);

  let pred: Tensor = logits.argmax(-1, false);

  let pred_labels: Vec<i64> = to_vec_i64(pred);

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

fn calculate_true_classes(x_idx: &Tensor) -> Tensor {
  // x_idx is [[128, 16], Int64] with random values between 0 and 9 inclusive
  // println!("x_idx: {x_idx}");

  let x_idx_float: Tensor = x_idx.to_kind(Kind::Float);

  // x_idx_float is [[128, 16], Float] with random values between 0. and 9.
  // println!("x_idx_float: {x_idx_float}");

  // Returns the sum of each row of the input tensor in the given dimension
  // https://docs.pytorch.org/docs/main/generated/torch.sum.html
  let x_sum: Tensor =
    x_idx_float.sum_dim_intlist([1].as_slice(), false, Kind::Float);

  // x_sum is [[128, Float] where the values are the values of the 16
  // println!("x_sum: {x_sum}");

  // The sum values are mapped to a class value by calculating the remainder
  let y: Tensor = x_sum.remainder(CLASS_COUNT as f64).to_kind(Kind::Int64);

  // y is [[128], Int64] where the values are between 0 and 4 inclusive
  // println!("y: {y}");

  y
}

fn to_vec_i64(tensor: Tensor) -> Vec<i64> {
  let y_cpu_tensor: Tensor = tensor.to_device(Device::Cpu);

  Vec::<i64>::try_from(y_cpu_tensor).unwrap()
}
