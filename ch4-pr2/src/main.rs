use ::ndarray::{
  Array, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr, concatenate, s,
};
use ::ndarray_rand::RandomExt;
use ::ndarray_rand::rand_distr::Uniform;
use ::ndarray_rand::rand_distr::uniform::Error;
use ::rayon::prelude::*;

type Matrix = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64>;

type View<'a> = ArrayBase<ViewRepr<&'a f64>, Dim<[usize; 2]>, f64>;

fn main() {
  const N_FULL: usize = 4;

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
  let matrix_dim: (usize, usize) = a0.dim();

  // TODO: check that both matrices are N squared or set the type if possible

  let n_full = matrix_dim.0;

  let n_half: usize = n_full / 2;

  let a0_00: View = a0.slice(s![
    0..n_half,
    0..n_half,
  ]);

  let a0_01: View = a0.slice(s![
    0..n_half,
    n_half..n_full,
  ]);

  let a0_10: View = a0.slice(s![
    n_half..n_full,
    0..n_half,
  ]);

  let a0_11: View = a0.slice(s![
    n_half..n_full,
    n_half..n_full,
  ]);

  let a1_00: View = a1.slice(s![
    0..n_half,
    0..n_half,
  ]);

  let a1_01: View = a1.slice(s![
    0..n_half,
    n_half..n_full,
  ]);

  let a1_10: View = a1.slice(s![
    n_half..n_full,
    0..n_half,
  ]);

  let a1_11: View = a1.slice(s![
    n_half..n_full,
    n_half..n_full,
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

  let a: [(View, View); 4] = [
    (a0_00, a1_00),
    (a0_01, a1_01),
    (a0_10, a1_10),
    (a0_11, a1_11),
  ];

  // TODO: Will this always be collected in the same order as the input?

  let b: Vec<(usize, Matrix)> = a
    .par_iter()
    .enumerate()
    .map(
      |(index, (a0, a1)): (usize, &(View, View))| -> (usize, Matrix) {
        (index, a0.dot(a1))
      },
    )
    .collect();

  println!("--- b");
  println!("{b:?}");

  let b_00: &Matrix = &(b.get(0).unwrap().1);

  let b_01: &Matrix = &(b.get(1).unwrap().1);

  let b_10: &Matrix = &(b.get(2).unwrap().1);

  let b_11: &Matrix = &(b.get(3).unwrap().1);

  println!("--- b_##");
  println!("{b_00:6.4}");
  println!("{b_01:6.4}");
  println!("{b_10:6.4}");
  println!("{b_11:6.4}");

  let c = concatenate!(Axis(1), *b_00, *b_01);
  let d = concatenate!(Axis(1), *b_10, *b_11);

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
    let a0: Matrix = array![
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

    let a1: Matrix = array![
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
