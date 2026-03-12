use ::tch::Tensor;

fn main() {
  let tensor = Tensor::f_from_slice(&[
    1, 2, 3,
  ])
  .expect("Failed to create tensor");

  println!("Tensor: {tensor:?}");
}
