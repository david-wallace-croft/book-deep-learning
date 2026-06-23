use ::std::io::Error;
use ch7_pr1::aliases::Dataset;
use ch7_pr1::loader::Loader;
use ch7_pr1::network::Network;

const EPOCH_COUNT: usize = 3;

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

  let mut network: Network = Network::new(&train_dataset);

  for epoch in 1..=EPOCH_COUNT {
    network.train(&train_dataset);

    println!(
      "Epoch {}: Loss = {:4}",
      epoch,
      network.total_loss / train_dataset.len() as f32
    );
  }

  let mut total_loss = 0.;

  let length = test_dataset.len();

  for (image, category) in test_dataset {
    // Loader::print_image(&image);

    let weight_index = 0;

    let y_pred: f32 = network.predict(&image, weight_index);

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
