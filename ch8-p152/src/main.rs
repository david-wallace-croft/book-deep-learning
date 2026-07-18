use self::sum_mod_transformer::SumModTransformer;
use ::tch::nn::{Adam, Optimizer, OptimizerConfig, Path, VarStore};
use ::tch::{Device, Kind, Result, Tensor};

mod encoder_block;
mod mhsa;
mod sum_mod_transformer;

const BATCH: i64 = 128;
const D_FF: i64 = 256;
const D_MODEL: i64 = 64;
const DROPOUT_P: f64 = 0.1;
const EPOCHS: i64 = 300;
const LEARNING_RATE: f64 = 1e-3;
const N_CLASSES: i64 = 5;
const N_HEADS: i64 = 4;
const N_LAYERS: i64 = 2;
const SEED: i64 = 42;
const T_STEPS: i64 = 16;
const VOCAB: i64 = 10;

fn main() -> Result<()> {
  tch::manual_seed(SEED);

  let device: Device = Device::cuda_if_available();

  let var_store: VarStore = VarStore::new(device);

  let root: &Path<'_> = &var_store.root();

  let model: SumModTransformer = SumModTransformer::new(
    root, VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, N_CLASSES, T_STEPS,
    DROPOUT_P, device,
  );

  let mut opt: Optimizer =
    Adam::default().build(&var_store, LEARNING_RATE).unwrap();

  for epoch in 1..=EPOCHS {
    let x_idx: Tensor = Tensor::randint(
      VOCAB,
      [
        BATCH, T_STEPS,
      ],
      (Kind::Int64, device),
    );

    let y: Tensor = x_idx
      .to_kind(Kind::Float)
      .sum_dim_intlist([1].as_slice(), false, Kind::Float)
      .remainder(N_CLASSES as f64)
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
    VOCAB,
    [
      test_b, T_STEPS,
    ],
    (Kind::Int64, device),
  );

  let y: Tensor = x_idx
    .to_kind(Kind::Float)
    .sum_dim_intlist([1].as_slice(), false, Kind::Float)
    .remainder(N_CLASSES as f64)
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
  let pred: Tensor = logits.argmax(-1, false);

  let correct: Tensor =
    pred.eq_tensor(y).to_kind(Kind::Float).mean(Kind::Float);

  correct.double_value(&[])
}
