use ::std::io::Error;
use ch7_pr1::aliases::Dataset;
use ch7_pr1::loader::Loader;
use ch7_pr1::network::Network;

const EPOCH_COUNT: usize = 3;

fn main() -> Result<(), Error> {
  let test_data_loader: Loader = Loader::default_test_data_loader();

  let mut test_dataset: Dataset = test_data_loader.load()?;

  test_dataset.drain(10..10_000);

  println!("Length of test dataset: {}", test_dataset.len());

  let train_data_loader: Loader = Loader::default_train_data_loader();

  let mut train_dataset: Dataset = train_data_loader.load()?;

  train_dataset.drain(10_000..60_000);

  println!("Length of train dataset: {}", train_dataset.len());

  let record_count = train_dataset.len();

  println!("Record count: {}", record_count);

  let mut network: Network = Network::new(&train_dataset);

  for epoch in 1..=EPOCH_COUNT {
    network.train(&train_dataset);

    println!(
      "Epoch {}: Loss = {:4}",
      epoch,
      network.total_loss / (train_dataset.len() * network.label_count) as f32
    );
  }

  let mut total_loss = 0.;

  let length = test_dataset.len();

  for (image, category) in test_dataset {
    // Loader::print_image(&image);

    let mut max = f32::MIN;

    let mut guess: usize = 0;

    for weight_index in 0..network.label_count {
      let y_pred: f32 = network.predict(&image, weight_index);

      if y_pred > max {
        max = y_pred;

        guess = weight_index;
      }

      print!("{weight_index}={y_pred:.6} ");

      let y_true = if category as usize == weight_index {
        1.
      } else {
        0.
      };

      // let loss: f32 = ::ch7_pr1::binary_cross_entropy(y_true, y_pred);

      let loss: f32 = (y_true - y_pred).abs();

      total_loss += loss;
    }

    println!("actual = {category} guess = {guess}");
  }

  println!(
    "Total loss for test dataset: {}",
    total_loss / (length * network.label_count) as f32
  );

  // println!("{:?}", network.fc_weight_matrix);

  // println!("{:?}", network.kernel);

  Ok(())
}
