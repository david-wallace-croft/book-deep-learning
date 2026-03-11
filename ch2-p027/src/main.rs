use ::linfa::Dataset;
use ::ndarray::array;

fn main() {
  let features = array![
    [
      1., 2.
    ],
    [
      3., 4.
    ]
  ];

  let targets = array![
    0, 1
  ];

  let dataset = Dataset::new(features, targets);

  println!("Linfa dataset: {dataset:?}");
}
