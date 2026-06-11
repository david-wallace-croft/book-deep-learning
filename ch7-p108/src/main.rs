use ::ndarray::prelude::*;
use ::ndarray::{OwnedRepr, ViewRepr};

fn main() {
  let input: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> = array![
    [
      1., 2., 3., 4.
    ],
    [
      5., 6., 7., 8.
    ],
    [
      9., 10., 11., 12.
    ],
    [
      13., 14., 15., 16.
    ],
  ];

  let kernel: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> = array![
    [
      1., 0.
    ],
    [
      0., -1.
    ],
  ];

  let output: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> =
    convolve2d(&input, &kernel);

  println!("{output}");
}

fn convolve2d(
  input: &Array2<f64>,
  kernel: &Array2<f64>,
) -> Array2<f64> {
  let (h, w): (usize, usize) = input.dim();

  let (k_h, k_w): (usize, usize) = kernel.dim();

  let out_h: usize = h - k_h + 1;

  let out_w: usize = w - k_w + 1;

  let mut output: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64> =
    Array2::<f64>::zeros((out_h, out_w));

  for i in 0..out_h {
    for j in 0..out_w {
      let window: ArrayBase<ViewRepr<&f64>, Dim<[usize; 2]>, f64> = input
        .slice(s![
          i..i + k_h,
          j..j + k_w
        ]);

      let sum: f64 = (&window * kernel).sum();

      output[(i, j)] = sum;
    }
  }

  output
}
