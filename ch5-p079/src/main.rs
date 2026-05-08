use ::autodiff::*;

fn main() {
  let input_size = 3;

  let hidden_size = 4;

  let mut w1 = vec![vec![0.1; input_size]; hidden_size];

  let mut b1 = vec![0.; hidden_size];

  let mut w2 = vec![0.1; hidden_size];

  let mut b2 = 0.;

  let data = vec![
    (
      vec![
        3., 5., 2.,
      ],
      5.,
    ),
    (
      vec![
        5., 7., 1.,
      ],
      7.,
    ),
    (
      vec![
        1., 9., 4.,
      ],
      9.,
    ),
    (
      vec![
        2., 3., 6.,
      ],
      6.,
    ),
    (
      vec![
        4., 1., 3.,
      ],
      4.,
    ),
    (
      vec![
        0., 8., 2.,
      ],
      8.,
    ),
    (
      vec![
        7., 2., 1.,
      ],
      7.,
    ),
    (
      vec![
        6., 3., 4.,
      ],
      6.,
    ),
    (
      vec![
        5., 0., 2.,
      ],
      5.,
    ),
    (
      vec![
        2., 6., 9.,
      ],
      9.,
    ),
  ];

  let lr = 0.001;

  let epochs = 1_000;

  for epoch in 0..epochs {
    let mut total_loss = 0.;

    for (x, target) in &data {
      let loss_fn = |params: &[FT<f64>]| {
        let mut idx = 0;

        let w1_ft: Vec<Vec<FT<f64>>> = (0..hidden_size)
          .map(|_| {
            (0..input_size)
              .map(|_| {
                let v = params[idx];

                idx += 1;

                v
              })
              .collect()
          })
          .collect();

        let b1_ft: Vec<FT<f64>> = (0..hidden_size)
          .map(|_| {
            let v = params[idx];

            idx += 1;

            v
          })
          .collect();

        let w2_ft: Vec<FT<f64>> = (0..hidden_size)
          .map(|_| {
            let v = params[idx];

            idx += 1;

            v
          })
          .collect();

        let b2_ft = params[idx];

        let h: Vec<FT<f64>> = (0..hidden_size)
          .map(|i| {
            let z =
              (0..input_size).map(|j| w1_ft[i][j] * x[j]).sum::<FT<f64>>()
                + b1_ft[i];

            relu(z)
          })
          .collect();

        let y_hat =
          (0..hidden_size).map(|i| w2_ft[i] * h[i]).sum::<FT<f64>>() + b2_ft;

        (y_hat - *target).powi(2)
      };

      let mut flat_params = vec![];

      for row in &w1 {
        flat_params.extend_from_slice(row);
      }

      flat_params.extend_from_slice(&b1);

      flat_params.extend_from_slice(&w2);

      flat_params.push(b2);

      let grads = grad(loss_fn, &flat_params);

      let input_ft: Vec<FT<f64>> =
        flat_params.iter().map(|&x| FT::cst(x)).collect();

      let loss = loss_fn(&input_ft);

      total_loss += loss.x;

      let mut idx = 0;

      for i in 0..hidden_size {
        for j in 0..input_size {
          w1[i][j] -= lr * grads[idx];

          idx += 1;
        }
      }

      for i in 0..hidden_size {
        b1[i] -= lr * grads[idx];

        idx += 1;
      }

      for i in 0..hidden_size {
        w2[i] -= lr * grads[idx];

        idx += 1;
      }

      b2 -= lr * grads[idx];
    }

    if epoch % 20 == 0 {
      println!("Epoch {epoch}: Loss = {:.5}", total_loss);
    }
  }

  let test_input = vec![
    0.7, 0.4, 1.,
  ];

  let hidden_out: Vec<f64> = (0..hidden_size)
    .map(|i| {
      let z: f64 = (0..input_size)
        .map(|j| w1[i][j] * test_input[j])
        .sum::<f64>()
        + b1[i];

      z.max(0.)
    })
    .collect();

  let y_pred: f64 =
    hidden_out.iter().zip(&w2).map(|(h, w)| h * w).sum::<f64>() + b2;

  println!("\n Test input: {:?}", test_input);

  println!("Prediction (network): {:.4}", y_pred);
  println!(
    "Ground truth (max): {:4}",
    test_input.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
  );
}

fn relu(x: FT<f64>) -> FT<f64> {
  if x.x > 0. {
    x
  } else {
    FT::cst(0.)
  }
}
