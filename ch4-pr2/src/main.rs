use ::ndarray::{Array, ArrayBase, Dim, OwnedRepr, ViewRepr, s};
use ::ndarray_rand::RandomExt;
use ::ndarray_rand::rand_distr::Uniform;
use ::ndarray_rand::rand_distr::uniform::Error;
use ndarray::{Axis, concatenate};

const N_FULL: usize = 4;
const N_HALF: usize = N_FULL / 2;

fn main() {
  let result: Result<Uniform<f64>, Error> = Uniform::new(0., 1.);

  let uniform: Uniform<f64> = result.unwrap();

  let a0: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> =
    Array::random((N_FULL, N_FULL), uniform);

  let a1: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> =
    Array::random((N_FULL, N_FULL), uniform);

  // println!("{a0:6.4}");

  // println!("{a1:6.4}");

  let a0_00: ArrayBase<ViewRepr<&f64>, _, f64> = a0.slice(s![
    0..N_HALF,
    0..N_HALF,
  ]);

  let a0_10: ArrayBase<ViewRepr<&f64>, _, f64> = a0.slice(s![
    N_HALF..N_FULL,
    0..N_HALF,
  ]);

  let a0_01: ArrayBase<ViewRepr<&f64>, _, f64> = a0.slice(s![
    0..N_HALF,
    N_HALF..N_FULL,
  ]);

  let a0_11: ArrayBase<ViewRepr<&f64>, _, f64> = a0.slice(s![
    N_HALF..N_FULL,
    N_HALF..N_FULL,
  ]);

  let a1_00: ArrayBase<ViewRepr<&f64>, _, f64> = a1.slice(s![
    0..N_HALF,
    0..N_HALF,
  ]);

  let a1_10: ArrayBase<ViewRepr<&f64>, _, f64> = a1.slice(s![
    N_HALF..N_FULL,
    0..N_HALF,
  ]);

  let a1_01: ArrayBase<ViewRepr<&f64>, _, f64> = a1.slice(s![
    0..N_HALF,
    N_HALF..N_FULL,
  ]);

  let a1_11: ArrayBase<ViewRepr<&f64>, _, f64> = a1.slice(s![
    N_HALF..N_FULL,
    N_HALF..N_FULL,
  ]);

  let b_00: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> = a0_00.dot(&a1_00);

  let b_01: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> = a0_01.dot(&a1_01);

  let b_10: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> = a0_10.dot(&a1_10);

  let b_11: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> = a0_11.dot(&a1_11);

  // TODO: Check that I am concatenating in the right order

  let c = concatenate!(Axis(0), b_00, b_01);

  let d = concatenate!(Axis(0), b_10, b_11);

  println!("{c:6.4}");

  println!("{d:6.4}");

  let e = concatenate!(Axis(1), c, d);

  println!("{e:6.4}");
}

// #[cfg(test)]
// mod test {
//   use super::*;

//   #[test]
//   fn test() {
//     // TODO
//   }
// }
