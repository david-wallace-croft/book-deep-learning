use ::ndarray::{
  Array, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr, concatenate, s,
};
use ::ndarray_rand::RandomExt;
use ::ndarray_rand::rand_distr::Uniform;
use ::ndarray_rand::rand_distr::uniform::Error;

// TODO: Make n a variable extracted from the inputs
const N_FULL: usize = 4;

const N_HALF: usize = N_FULL / 2;

type Matrix = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64>;

type View<'a> = ArrayBase<ViewRepr<&'a f64>, Dim<[usize; 2]>, f64>;

fn main() {
  let result: Result<Uniform<f64>, Error> = Uniform::new(0., 1.);

  let uniform: Uniform<f64> = result.unwrap();

  let a0: Matrix = Array::random((N_FULL, N_FULL), uniform);

  let a1: Matrix = Array::random((N_FULL, N_FULL), uniform);

  println!("--- a0");
  println!("{a0:6.4}");

  println!("--- a1");
  println!("{a1:6.4}");

  let e = problem_4_2(a0, a1);

  println!("--- e");
  println!("{e:6.4}");
}

fn problem_4_2(
  a0: Matrix,
  a1: Matrix,
) -> Matrix {
  let a0_00: View = a0.slice(s![
    0..N_HALF,
    0..N_HALF,
  ]);

  let a0_01: View = a0.slice(s![
    0..N_HALF,
    N_HALF..N_FULL,
  ]);

  let a0_10: View = a0.slice(s![
    N_HALF..N_FULL,
    0..N_HALF,
  ]);

  let a0_11: View = a0.slice(s![
    N_HALF..N_FULL,
    N_HALF..N_FULL,
  ]);

  let a1_00: View = a1.slice(s![
    0..N_HALF,
    0..N_HALF,
  ]);

  let a1_01: View = a1.slice(s![
    0..N_HALF,
    N_HALF..N_FULL,
  ]);

  let a1_10: View = a1.slice(s![
    N_HALF..N_FULL,
    0..N_HALF,
  ]);

  let a1_11: View = a1.slice(s![
    N_HALF..N_FULL,
    N_HALF..N_FULL,
  ]);

  println!("--- a0_##");
  println!("{a0_00:6.4}");
  println!("{a0_01:6.4}");
  println!("{a0_10:6.4}");
  println!("{a0_11:6.4}");
  println!("--- a1_##");
  println!("{a1_00:6.4}");
  println!("{a1_01:6.4}");
  println!("{a1_10:6.4}");
  println!("{a1_11:6.4}");

  let b_00: Matrix = a0_00.dot(&a1_00);

  let b_01: Matrix = a0_01.dot(&a1_01);

  let b_10: Matrix = a0_10.dot(&a1_10);

  let b_11: Matrix = a0_11.dot(&a1_11);

  println!("--- b_##");
  println!("{b_00:6.4}");
  println!("{b_01:6.4}");
  println!("{b_10:6.4}");
  println!("{b_11:6.4}");

  let c = concatenate!(Axis(1), b_00, b_01);
  let d = concatenate!(Axis(1), b_10, b_11);

  println!("--- c");
  println!("{c:6.4}");
  println!("--- d");
  println!("{d:6.4}");

  concatenate!(Axis(0), c, d)
}

#[cfg(test)]
mod test {
  use super::*;
  use ::ndarray::array;

  #[test]
  fn test() {
    let a0 = array![
      [
        0., 0., 1., 1.,
      ],
      [
        2., 2., 3., 3.,
      ],
      [
        4., 4., 5., 5.
      ],
      [
        6., 6., 7., 7.
      ]
    ];

    let a1 = array![
      [
        0., 0., 1., 1.,
      ],
      [
        2., 2., 3., 3.,
      ],
      [
        4., 4., 5., 5.
      ],
      [
        6., 6., 7., 7.
      ]
    ];

    let expected: Matrix = array![
      [
        0.0000, 0.0000, 4.0000, 4.0000
      ],
      [
        4.0000, 4.0000, 12.0000, 12.0000
      ],
      [
        40.0000, 40.0000, 60.0000, 60.0000
      ],
      [
        60.0000, 60.0000, 84.0000, 84.0000
      ]
    ];

    let actual: Matrix = problem_4_2(a0, a1);

    assert_eq!(actual, expected);
  }
}
