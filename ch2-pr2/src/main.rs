use ::ndarray::Array;
use ::ndarray::ArrayBase;
use ::ndarray::Dim;
use ::ndarray::OwnedRepr;

fn main() {
  let arr: ArrayBase<OwnedRepr<i32>, Dim<[usize; 1]>> = Array::from_vec(vec![
    5, 10, 15,
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
        5, 10, 15,
      ]);

    let actual: &String = &format!("{arr:?}");

    assert_eq!(
      actual,
      "[5, 10, 15], shape=[3], strides=[1], layout=CFcf (0xf), const ndim=1"
    );
  }
}
