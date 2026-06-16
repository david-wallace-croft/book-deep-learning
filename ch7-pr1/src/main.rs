use self::data::Data;
use self::loader::Loader;
use ::std::io::Error;
use ::std::{iter::FlatMap, slice::Iter};

mod data;
mod loader;

type Matrix = Vec<Vec<f32>>;

fn main() -> Result<(), Error> {
  let loader: Loader = Loader::default();

  let data: Data = loader.load()?;

  let record_count = data.train_images.len();

  println!("Record count: {}", record_count);

  let record_count = data.train_labels.len();

  println!("Record count: {}", record_count);

  let cat_image_matrix: Matrix = vec![
    vec![
      0., 1., 1., 0., 2., 3.,
    ],
    vec![
      1., 2., 0., 1., 3., 1.,
    ],
    vec![
      0., 1., 1., 0., 2., 2.,
    ],
    vec![
      1., 0., 2., 3., 1., 0.,
    ],
    vec![
      2., 3., 1., 0., 1., 2.,
    ],
    vec![
      0., 1., 0., 2., 3., 1.,
    ],
  ];

  let non_cat_image_matrix: Matrix = vec![
    vec![
      1., 0., 0., 1., 0., 0.,
    ],
    vec![
      0., 1., 0., 0., 1., 0.,
    ],
    vec![
      1., 0., 1., 0., 0., 1.,
    ],
    vec![
      0., 1., 0., 1., 0., 0.,
    ],
    vec![
      0., 0., 1., 0., 1., 0.,
    ],
    vec![
      1., 0., 0., 1., 0., 1.,
    ],
  ];

  let new_image: Matrix = vec![
    vec![
      0., 1., 1., 0., 2., 3.,
    ],
    vec![
      1., 2., 0., 1., 3., 1.,
    ],
    vec![
      0., 1., 1., 0., 2., 2.,
    ],
    vec![
      1., 0., 2., 3., 1., 0.,
    ],
    vec![
      2., 3., 1., 0., 1., 2.,
    ],
    vec![
      0., 1., 0., 2., 3., 1.,
    ],
  ];

  let dataset: Vec<(Matrix, f32)> = vec![
    (cat_image_matrix, 1.),
    (non_cat_image_matrix, 0.),
  ];

  let mut kernel: Matrix = vec![
    vec![
      0.1, 0.2, -0.1,
    ],
    vec![
      0.0, 0.0, 0.1,
    ],
    vec![
      -0.2, 0., 0.2,
    ],
  ];

  let temp_conv: Matrix = conv2d(&dataset[0].0, &kernel);

  let (temp_pool, _): (Matrix, Vec<Vec<(usize, usize)>>) =
    max_pool2x2(&temp_conv);

  let flat_len: usize = temp_pool.len() * temp_pool[0].len();

  let mut fc_weights: Vec<f32> = vec![0.5; flat_len];

  let mut fc_bias: f32 = 0.;

  let lr: f32 = 0.01;

  for epoch in 0..50 {
    let mut total_loss: f32 = 0.;

    for (image, label) in &dataset {
      let mut conv_out: Matrix = conv2d(image, &kernel);

      for i in 0..conv_out.len() {
        for j in 0..conv_out[0].len() {
          conv_out[i][j] = relu(conv_out[i][j]);
        }
      }

      let (pool_out, max_pos): (Matrix, Vec<Vec<(usize, usize)>>) =
        max_pool2x2(&conv_out);

      let flat: Vec<f32> = flatten(&pool_out);

      let z: f32 = flat
        .iter()
        .zip(fc_weights.iter())
        .map(|(x, w)| x * w)
        .sum::<f32>()
        + fc_bias;

      let y_pred: f32 = sigmoid(z);

      let loss: f32 = binary_cross_entropy(*label, y_pred);

      total_loss += loss;

      let dz: f32 = y_pred - label;

      for i in 0..fc_weights.len() {
        fc_weights[i] -= lr * dz * flat[i];
      }

      fc_bias -= lr * dz;

      let mut d_pool_out: Matrix =
        vec![vec![0.; pool_out[0].len()]; pool_out.len()];

      let mut idx: usize = 0;

      #[expect(clippy::needless_range_loop)]
      for i in 0..pool_out.len() {
        for j in 0..pool_out[0].len() {
          d_pool_out[i][j] = dz * fc_weights[idx];

          idx += 1;
        }
      }

      let d_conv_out: Matrix = max_pool2x2_backprop(
        &d_pool_out,
        &max_pos,
        conv_out.len(),
        conv_out[0].len(),
      );

      let mut d_conv_out_relu: Matrix =
        vec![vec![0.; conv_out[0].len()]; conv_out.len()];

      for i in 0..conv_out.len() {
        for j in 0..conv_out[0].len() {
          d_conv_out_relu[i][j] = d_conv_out[i][j] * relu_deriv(conv_out[i][j]);
        }
      }

      conv2d_backprop(&d_conv_out_relu, image, &mut kernel, lr);
    }

    println!(
      "Epoch {}: Loss = {:4}",
      epoch,
      total_loss / dataset.len() as f32
    );
  }

  println!("Trained kernel: {kernel:?}");

  println!("Trained FC weights: {fc_weights:?}");

  let prob: f32 = predict(&new_image, &kernel, &fc_weights, fc_bias);

  if prob >= 0.5 {
    println!("Prediction: Cat ({:.2}%)", prob * 100.);
  } else {
    println!("Prediction: Not Cat ({:.2}%)", prob * 100.);
  }

  Ok(())
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

fn conv2d_backprop(
  d_out: &[Vec<f32>],
  input: &[Vec<f32>],
  kernel: &mut [Vec<f32>],
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

fn flatten(matrix: &[Vec<f32>]) -> Vec<f32> {
  let row_iter: Iter<'_, Vec<f32>> = matrix.iter();

  let flat_map: FlatMap<Iter<'_, Vec<f32>>, Vec<f32>, _> =
    row_iter.flat_map(|row: &Vec<f32>| row.clone());

  flat_map.collect()
}

fn max_pool2x2(input: &[Vec<f32>]) -> (Matrix, Vec<Vec<(usize, usize)>>) {
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

fn max_pool2x2_backprop(
  d_out: &[Vec<f32>],
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

fn predict(
  image: &[Vec<f32>],
  kernel: &[Vec<f32>],
  fc_weights: &[f32],
  fc_bias: f32,
) -> f32 {
  let mut conv_out: Matrix = conv2d(image, kernel);

  for row in conv_out.iter_mut() {
    for val in row.iter_mut() {
      *val = relu(*val);
    }
  }

  let (pool_out, _): (Matrix, Vec<Vec<(usize, usize)>>) =
    max_pool2x2(&conv_out);

  let flat: Vec<f32> = flatten(&pool_out);

  let z: f32 = flat
    .iter()
    .zip(fc_weights.iter())
    .map(|(x, w)| x * w)
    .sum::<f32>()
    + fc_bias;

  sigmoid(z)
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
