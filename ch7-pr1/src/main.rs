use ::std::io::Error;
use ch7_pr1::aliases::{Dataset, Matrix, Vector};
use ch7_pr1::loader::Loader;

const EPOCHS: usize = 3;

fn main() -> Result<(), Error> {
  let test_data_loader: Loader = Loader::default_test_data_loader();

  let test_dataset: Dataset = test_data_loader.load()?;

  println!("Length of test dataset: {}", test_dataset.len());

  let train_data_loader: Loader = Loader::default_train_data_loader();

  let train_dataset: Dataset = train_data_loader.load()?;

  println!("Length of train dataset: {}", train_dataset.len());

  // Temporary hack: training to recognize when the category is a 1.
  let train_dataset: Dataset = train_dataset
    .into_iter()
    .map(|(image, category)| {
      (
        image,
        if category != 3 {
          0
        } else {
          category
        },
      )
    })
    .collect();

  let record_count = train_dataset.len();

  println!("Record count: {}", record_count);

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

  let temp_conv: Matrix = ::ch7_pr1::conv2d(&train_dataset[0].0, &kernel);

  let (temp_pool, _): (Matrix, Vec<Vec<(usize, usize)>>) =
    ::ch7_pr1::max_pool2x2(&temp_conv);

  let flat_len: usize = temp_pool.len() * temp_pool[0].len();

  let mut fc_weights: Vector = vec![0.5; flat_len];

  let mut fc_bias: f32 = 0.;

  let lr: f32 = 0.001;

  for epoch in 0..EPOCHS {
    let mut total_loss: f32 = 0.;

    for (index, (image, label)) in train_dataset.iter().enumerate() {
      if index % 10_000 == 0 {
        println!("index = {index}");
      }

      // println!("{image:?}");

      let mut conv_out: Matrix = ::ch7_pr1::conv2d(image, &kernel);

      // println!("{conv_out:?}");

      for i in 0..conv_out.len() {
        for j in 0..conv_out[0].len() {
          conv_out[i][j] = ::ch7_pr1::relu(conv_out[i][j]);
        }
      }

      // println!("{conv_out:?}");

      let (pool_out, max_pos): (Matrix, Vec<Vec<(usize, usize)>>) =
        ::ch7_pr1::max_pool2x2(&conv_out);

      let flat: Vector = ::ch7_pr1::flatten(&pool_out);

      // println!("flat {flat:?}");

      let z: f32 = flat
        .iter()
        .zip(fc_weights.iter())
        .map(|(x, w)| x * w)
        .sum::<f32>()
        + fc_bias;

      if z.is_nan() {
        panic!("z is NaN at index {index}");
      }

      // println!("z {z}");

      let y_pred: f32 = ::ch7_pr1::sigmoid(z);

      // println!("y_pred {y_pred}");

      let y_true = if *label == 3 {
        1.
      } else {
        0.
      };

      let loss: f32 = ::ch7_pr1::binary_cross_entropy(y_true, y_pred);

      // println!("y_pred {y_pred} label {label}");

      // let loss: f32 = (y_pred - label).abs();

      total_loss += loss;

      let dz: f32 = y_pred - y_true;

      for i in 0..fc_weights.len() {
        fc_weights[i] -= lr * dz * flat[i];
      }

      // println!("fc_weights {fc_weights:?}");

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

      let d_conv_out: Matrix = ::ch7_pr1::max_pool2x2_backprop(
        &d_pool_out,
        &max_pos,
        conv_out.len(),
        conv_out[0].len(),
      );

      // println!("d_conv_out {d_conv_out:?}");

      let mut d_conv_out_relu: Matrix =
        vec![vec![0.; conv_out[0].len()]; conv_out.len()];

      for i in 0..conv_out.len() {
        for j in 0..conv_out[0].len() {
          d_conv_out_relu[i][j] =
            d_conv_out[i][j] * ::ch7_pr1::relu_deriv(conv_out[i][j]);
        }
      }

      // println!("d_conv_out_relu {d_conv_out_relu:?}");

      ::ch7_pr1::conv2d_backprop(&d_conv_out_relu, image, &mut kernel, lr);
    }

    println!(
      "Epoch {}: Loss = {:4}",
      epoch,
      total_loss / train_dataset.len() as f32
    );
  }

  // println!("Trained kernel: {kernel:?}");

  // println!("Trained FC weights: {fc_weights:?}");

  let mut total_loss = 0.;

  let length = test_dataset.len();

  for (image, category) in test_dataset {
    // Loader::print_image(&image);

    let y_pred: f32 = ::ch7_pr1::predict(&image, &kernel, &fc_weights, fc_bias);

    let y_true = if category == 3 {
      1.
    } else {
      0.
    };

    let loss: f32 = ::ch7_pr1::binary_cross_entropy(y_true, y_pred);

    total_loss += loss;

    // if prob >= 0.5 {
    //   println!("Prediction: Three ({:.2}%)", prob * 100.);
    // } else {
    //   println!("Prediction: Not Three ({:.2}%)", prob * 100.);
    // }
  }

  println!(
    "Total loss for test dataset: {}",
    total_loss / length as f32
  );

  Ok(())
}
