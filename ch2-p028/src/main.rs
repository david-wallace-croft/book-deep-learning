use ::tch::Tensor;

fn main() {
  let tensor = Tensor::f_from_slice(&[
    1, 2, 3,
  ])
  .expect("Failed to create tensor");

  println!("Tensor: {tensor:?}");
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
    let tensor = Tensor::f_from_slice(&[
      1, 2, 3,
    ])
    .expect("Failed to create tensor");

    let actual = &format!("{tensor:?}");

    assert_eq!(actual, "[1, 2, 3]");
  }
}
