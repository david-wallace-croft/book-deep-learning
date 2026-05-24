#![expect(dead_code)]

fn main() {
  todo!()
}

fn binary_cross_entropy(
  y_true: f32,
  y_pred: f32,
) -> f32 {
  -(y_true * y_pred.ln() + (1. - y_true) * (1. - y_pred).ln())
}

fn conv2d(
  input: &[Vec<f32>],
  kernel: &[Vec<f32>],
) -> Vec<Vec<f32>> {
  let h = input.len();

  let w = input[0].len();

  let kh = kernel.len();

  let kw = kernel[0].len();

  let mut output = vec![vec![0.; w - kw + 1]; h - kh + 1];

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

fn conv2d_backprop(
  d_out: &[Vec<f32>],
  input: &[Vec<f32>],
  kernel: &mut [Vec<f32>],
  lr: f32,
) {
  let kh = kernel.len();

  let kw = kernel[0].len();

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

fn relu(x: f32) -> f32 {
  if x > 0. {
    x
  } else {
    0.
  }
}

fn relu_deriv(x: f32) -> f32 {
  if x > 0. {
    1.
  } else {
    0.
  }
}

fn sigmoid(x: f32) -> f32 {
  1. / (1. + (-x).exp())
}
