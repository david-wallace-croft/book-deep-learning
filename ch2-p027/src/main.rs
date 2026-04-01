use ::linfa::Dataset;
use ::linfa::DatasetBase;
use ::ndarray::ArrayBase;
use ::ndarray::Dim;
use ::ndarray::OwnedRepr;
use ::ndarray::array;

type Features = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
type Targets = ArrayBase<OwnedRepr<i32>, Dim<[usize; 1]>>;

fn main() {
  let features: Features = array![
    [
      1., 2.
    ],
    [
      3., 4.
    ]
  ];

  let targets: Targets = array![
    0, 1
  ];

  let dataset: DatasetBase<Features, Targets> = Dataset::new(features, targets);

  println!("Linfa dataset: {dataset:?}");
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
    let features: Features = array![
      [
        1., 2.
      ],
      [
        3., 4.
      ]
    ];

    let targets: Targets = array![
      0, 1
    ];

    let dataset: DatasetBase<Features, Targets> =
      Dataset::new(features, targets);

    let actual: &String = &format!("{dataset:?}");

    assert_eq!(
      actual,
      "DatasetBase { records: [[1.0, 2.0],\n [3.0, 4.0]], shape=[2, 2], \
      strides=[2, 1], layout=Cc (0x5), const ndim=2, targets: [0, 1], \
      shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1, weights: [], \
      shape=[0], strides=[0], layout=CFcf (0xf), const ndim=1, \
      feature_names: [], target_names: [] }"
    );
  }
}
