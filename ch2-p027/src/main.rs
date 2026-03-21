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

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
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

    let actual = &format!("{dataset:?}");

    assert_eq!(
      actual,
      "DatasetBase { records: [[1.0, 2.0],\n [3.0, 4.0]], shape=[2, 2], \
      strides=[2, 1], layout=Cc (0x5), const ndim=2, targets: [0, 1], \
      shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1, weights: [], \
      shape=[0], strides=[0], layout=CFcf (0xf), const ndim=1, \
      feature_names: [] }"
    );
  }
}
