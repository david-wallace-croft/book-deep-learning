use ::ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use ::ndarray_rand::RandomExt;
use ::ndarray_rand::rand_distr::Uniform;
use ::ndarray_rand::rand_distr::uniform::Error;

fn main() {
  let result: Result<Uniform<f64>, Error> = Uniform::new(0., 1.);

  let uniform: Uniform<f64> = result.unwrap();

  let array_base: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> =
    Array::random((100, 100), uniform);

  println!("{:6.4}", array_base);
}

// #[cfg(test)]
// mod test {
//   use super::*;

//   #[test]
//   fn test() {
//     // TODO
//   }
// }
