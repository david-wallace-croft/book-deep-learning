#![expect(dead_code)]
#![expect(unused_variables)]

fn main() {
  let cat_image_matrix = vec![
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

  let non_cat_image_matrix = vec![
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

  let dataset = vec![
    (cat_image_matrix, 1.),
    (non_cat_image_matrix, 0.),
  ];

  #[expect(unused_mut)]
  let mut kernel = vec![
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

  let temp_conv = conv2d(&dataset[0].0, &kernel);

  let (temp_pool, _) = max_pool2x2(&temp_conv);

  let flat_len = temp_pool.len() * temp_pool[0].len();

  #[expect(unused_mut)]
  let mut fc_weights = vec![0.5; flat_len];

  #[expect(unused_mut)]
  let mut fc_bias = 0.;

  let lr = 0.01;

  for epoch in 0..50 {
    #[expect(unused_mut)]
    let mut total_loss = 0.;

    for (image, label) in &dataset {
      todo!()
    }
  }

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

#[expect(clippy::type_complexity)]
fn max_pool2x2(
  input: &[Vec<f32>]
) -> (Vec<Vec<f32>>, Vec<Vec<(usize, usize)>>) {
  let h = input.len() / 2;

  let w = input[0].len() / 2;

  let mut output = vec![vec![0.; w]; h];

  let mut max_pos = vec![vec![(0, 0); w]; h];

  for i in 0..h {
    for j in 0..w {
      let mut max_val = f32::MIN;

      let mut pos = (0, 0);

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
) -> Vec<Vec<f32>> {
  let mut d_input = vec![vec![0.; w]; h];

  for i in 0..d_out.len() {
    for j in 0..d_out[0].len() {
      let (mi, mj) = max_pos[i][j];

      d_input[mi][mj] = d_out[i][j];
    }
  }

  d_input
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
