use ::tch::nn::{
  self, Adam, ModuleT, Optimizer, OptimizerConfig, Path, Sequential, VarStore,
};
use ::tch::{Device, Kind, TchError, Tensor};

const BATCH: i64 = 128;
const ITERS: i64 = 2_000;
const LEARNING_RATE: f64 = 2e-4;
const PRINT_EVERY: i64 = 100;
const SEED: i64 = 42;
const Z_DIM: i64 = 8;

fn main() -> Result<(), TchError> {
  tch::manual_seed(SEED);

  let device: Device = Device::cuda_if_available();

  let var_store_discriminator: VarStore = VarStore::new(device);

  let var_store_generator: VarStore = VarStore::new(device);

  let d: Sequential = build_discriminator(&var_store_discriminator.root());

  let g: Sequential = build_generator(&var_store_generator.root(), Z_DIM);

  let mut optimizer_discriminator: Optimizer =
    Adam::default().build(&var_store_discriminator, LEARNING_RATE)?;

  let mut optimizer_generator: Optimizer =
    Adam::default().build(&var_store_generator, LEARNING_RATE)?;

  for step in 1..=ITERS {
    let x_real: Tensor = sample_real(BATCH, device)?;

    let z: Tensor = Tensor::randn(
      [
        BATCH, Z_DIM,
      ],
      (Kind::Float, device),
    );

    let x_fake: Tensor = g.forward_t(&z, true);

    let d_real_logits: Tensor = d.forward_t(&x_real, true);

    let d_fake_logits: Tensor = d.forward_t(&x_fake.detach(), true);

    let loss_d_real: Tensor = -d_real_logits.log_sigmoid().mean(Kind::Float);

    let loss_d_fake: Tensor =
      -(-&d_fake_logits).log_sigmoid().mean(Kind::Float);

    let loss_d: Tensor = &loss_d_real + &loss_d_fake;

    optimizer_discriminator.backward_step(&loss_d);

    let z2: Tensor = Tensor::randn(
      [
        BATCH, Z_DIM,
      ],
      (Kind::Float, device),
    );

    let x_fake2: Tensor = g.forward_t(&z2, true);

    let d_fake2_logits: Tensor = d.forward_t(&x_fake2, true);

    let loss_g: Tensor = -d_fake2_logits.log_sigmoid().mean(Kind::Float);

    optimizer_generator.backward_step(&loss_g);

    if step % PRINT_EVERY == 0 {
      let ld: f64 = loss_d.to_device(Device::Cpu).double_value(&[]);

      let lg: f64 = loss_g.to_device(Device::Cpu).double_value(&[]);

      println!("step {step:4} | d_loss {ld:.4} | g_loss {lg:.4}");
    }
  }

  let z: Tensor = Tensor::randn(
    [
      10, Z_DIM,
    ],
    (Kind::Float, device),
  );

  let samples: Tensor = g.forward_t(&z, false).to_device(Device::Cpu);

  let flat: Vec<f32> = samples.view([-1]).try_into().unwrap();

  println!("generated samples (x,y):");

  for i in 0..10 {
    let x: f32 = flat[2 * i];

    let y: f32 = flat[2 * i + 1];

    println!("  {i:>2}: [{x:.3}, {y:.3}]");
  }

  Ok(())
}

fn build_discriminator(vs: &Path) -> Sequential {
  nn::seq()
    .add(nn::linear(vs / "d1", 2, 64, Default::default()))
    .add_fn(|xs: &Tensor| xs.leaky_relu())
    .add(nn::linear(vs / "d2", 64, 1, Default::default()))
}

fn build_generator(
  vs: &Path,
  z_dim: i64,
) -> Sequential {
  nn::seq()
    .add(nn::linear(vs / "g1", z_dim, 64, Default::default()))
    .add_fn(|xs: &Tensor| xs.relu())
    .add(nn::linear(vs / "g2", 64, 2, Default::default()))
}

fn sample_real(
  batch: i64,
  device: Device,
) -> Result<Tensor, TchError> {
  let half: i64 = batch / 2;

  let std: f64 = 0.5_f64;

  let mean1 = Tensor::f_from_slice(&[
    -2.0_f32, 0.,
  ])?
  .to_device(device)
  .view([
    1, 2,
  ]);

  let mean2: Tensor = Tensor::f_from_slice(&[
    2.0_f32, 0.,
  ])?
  .to_device(device)
  .view([
    1, 2,
  ]);

  let x1: Tensor = Tensor::randn(
    [
      half, 2,
    ],
    (Kind::Float, device),
  ) * std
    + &mean1;

  let x2: Tensor = Tensor::randn(
    [
      batch - half,
      2,
    ],
    (Kind::Float, device),
  ) * std
    + &mean2;

  Ok(Tensor::cat(
    &[
      x1, x2,
    ],
    0,
  ))
}
