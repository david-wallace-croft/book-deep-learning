use ::ndarray::{Array, ArrayBase, Dim, OwnedRepr, ViewRepr, s};
use ::ndarray_rand::RandomExt;
use ::ndarray_rand::rand_distr::Uniform;
use ::ndarray_rand::rand_distr::uniform::Error;

const N_FULL: usize = 4;
const N_HALF: usize = N_FULL / 2;

fn main() {
  let result: Result<Uniform<f64>, Error> = Uniform::new(0., 1.);

  let uniform: Uniform<f64> = result.unwrap();

  let array_base_0: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> =
    Array::random((N_FULL, N_FULL), uniform);

  // let array_base_1: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> =
  //   Array::random((N_FULL, N_FULL), uniform);

  // println!("{:6.4}", array_base_0);

  // println!("{:6.4}", array_base_1);

  let c0: ArrayBase<ViewRepr<&f64>, _, f64> = array_base_0.slice(s![
    0..N_HALF,
    0..N_HALF,
  ]);

  let c1: ArrayBase<ViewRepr<&f64>, _, f64> = array_base_0.slice(s![
    N_HALF..N_FULL,
    0..N_HALF,
  ]);

  // let c2: ArrayBase<ViewRepr<&f64>, _, f64> = array_base_0.slice(s![
  //   0..N_HALF,
  //   N_HALF..N_FULL,
  // ]);

  // let c3: ArrayBase<ViewRepr<&f64>, _, f64> = array_base_0.slice(s![
  //   N_HALF..N_FULL,
  //   N_HALF..N_FULL,
  // ]);

  // println!("{:6.4}", c0);
  // println!("{:6.4}", c1);
  // println!("{:6.4}", c2);
  // println!("{:6.4}", c3);

  let product: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> = c0.dot(&c1);

  println!("{product:6.4}");
}

// #[cfg(test)]
// mod test {
//   use super::*;

//   #[test]
//   fn test() {
//     // TODO
//   }
// }
