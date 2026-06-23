use self::aliases::{Matrix, Vector};
use ::std::iter::FlatMap;
use ::std::slice::Iter;

pub mod aliases;
pub mod loader;
pub mod network;

pub fn binary_cross_entropy(
  y_true: f32,
  y_pred: f32,
) -> f32 {
  -(y_true * y_pred.ln() + (1. - y_true) * (1. - y_pred).ln())
}

pub fn conv2d(
  input: &[Vector],
  kernel: &[Vector],
) -> Matrix {
  let h: usize = input.len();

  let w: usize = input[0].len();

  let kh: usize = kernel.len();

  let kw: usize = kernel[0].len();

  let mut output: Matrix = vec![vec![0.; w - kw + 1]; h - kh + 1];

  for i in 0..(h - kh + 1) {
    for j in 0..(w - kw + 1) {
      let mut sum = 0.;

      for m in 0..kh {
        for n in 0..kw {
          sum += input[i + m][j + n] * kernel[m][n];
        }
      }

      output[i][j] = sum;
    }
  }

  output
}

pub fn conv2d_backprop(
  d_out: &[Vector],
  input: &[Vector],
  kernel: &mut [Vector],
  lr: f32,
) {
  let kh: usize = kernel.len();

  let kw: usize = kernel[0].len();

  for m in 0..kh {
    for n in 0..kw {
      let mut grad = 0.;

      for i in 0..d_out.len() {
        for j in 0..d_out[0].len() {
          grad += input[i + m][j + n] * d_out[i][j];
        }
      }

      kernel[m][n] -= lr * grad;
    }
  }
}

pub fn flatten(matrix: &[Vector]) -> Vector {
  let row_iter: Iter<'_, Vector> = matrix.iter();

  let flat_map: FlatMap<Iter<'_, Vector>, Vector, _> =
    row_iter.flat_map(|row: &Vector| row.clone());

  flat_map.collect()
}

pub fn max_pool2x2(input: &[Vector]) -> (Matrix, Vec<Vec<(usize, usize)>>) {
  let h: usize = input.len() / 2;

  let w: usize = input[0].len() / 2;

  let mut output: Matrix = vec![vec![0.; w]; h];

  let mut max_pos: Vec<Vec<(usize, usize)>> = vec![vec![(0, 0); w]; h];

  for i in 0..h {
    for j in 0..w {
      let mut max_val = f32::MIN;

      let mut pos: (usize, usize) = (0, 0);

      for m in 0..2 {
        for n in 0..2 {
          let val = input[i * 2 + m][j * 2 + n];

          if val > max_val {
            max_val = val;

            pos = (i * 2 + m, j * 2 + n);
          }
        }
      }

      output[i][j] = max_val;

      max_pos[i][j] = pos;
    }
  }

  (output, max_pos)
}

pub fn max_pool2x2_backprop(
  d_out: &[Vector],
  max_pos: &[Vec<(usize, usize)>],
  h: usize,
  w: usize,
) -> Matrix {
  let mut d_input: Matrix = vec![vec![0.; w]; h];

  for i in 0..d_out.len() {
    for j in 0..d_out[0].len() {
      let (mi, mj): (usize, usize) = max_pos[i][j];

      d_input[mi][mj] = d_out[i][j];
    }
  }

  d_input
}

pub fn net_input(
  inputs: &[f32],
  weights: &[f32],
  bias: f32,
) -> f32 {
  inputs
    .iter()
    .zip(weights.iter())
    .map(|(x, w)| x * w)
    .sum::<f32>()
    + bias
}

pub fn perceptron(
  inputs: &[f32],
  weights: &[f32],
  bias: f32,
) -> f32 {
  let z: f32 = net_input(inputs, weights, bias);

  sigmoid(z)
}

pub fn relu(x: f32) -> f32 {
  if x > 0. {
    x
  } else {
    0.
  }
}

pub fn relu_deriv(x: f32) -> f32 {
  if x > 0. {
    1.
  } else {
    0.
  }
}

pub fn sigmoid(x: f32) -> f32 {
  1. / (1. + (-x).exp())
}
