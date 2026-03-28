use ::tch::TchError;
use ::tch::Tensor;

fn main() {
  let result: Result<Tensor, TchError> = Tensor::f_from_slice(&[
    1, 2, 3,
  ]);

  let tensor: Tensor = result.expect("Failed to create tensor");

  println!("Tensor: {tensor:?}");
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
    let result: Result<Tensor, TchError> = Tensor::f_from_slice(&[
      1, 2, 3,
    ]);

    let tensor: Tensor = result.expect("Failed to create tensor");

    let actual: &String = &format!("{tensor:?}");

    assert_eq!(actual, "[1, 2, 3]");
  }
}
