use ::ndarray::Array;
use ::ndarray::ArrayBase;
use ::ndarray::Dim;
use ::ndarray::OwnedRepr;

fn main() {
  let arr: ArrayBase<OwnedRepr<i32>, Dim<[usize; 1]>> = Array::from_vec(vec![
    1, 2, 3,
  ]);

  println!("Array: {arr:?}");
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
    let arr: ArrayBase<OwnedRepr<i32>, Dim<[usize; 1]>> =
      Array::from_vec(vec![
        1, 2, 3,
      ]);

    let actual: &String = &format!("{arr:?}");

    assert_eq!(
      actual,
      "[1, 2, 3], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1"
    );
  }
}
