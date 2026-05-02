const INPUTS: [(f64, f64); 4] = [
  (0., 0.),
  (0., 1.),
  (1., 0.),
  (1., 1.),
];

const WEIGHTS_AND: [f64; 3] = [
  -1.5, 1., 1.,
];

const WEIGHTS_NAND: [f64; 3] = [
  1., -0.75, -0.75,
];

const WEIGHTS_OR: [f64; 3] = [
  -0.5, 1., 1.,
];

fn main() {
  for (x1, x2) in INPUTS {
    let x = [
      1., x1, x2,
    ];

    let output = neural_network_xor(x);

    println!("Input: ({x1}, {x2}) => XOR: {output}");
  }
}

fn dot_product(
  w: [f64; 3],
  x: [f64; 3],
) -> f64 {
  w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
}

fn neural_network_xor(x: [f64; 3]) -> f64 {
  let y1: f64 = perceptron_nand(x);

  let y2: f64 = perceptron_or(x);

  let y: [f64; 3] = [
    1., y1, y2,
  ];

  perceptron_and(y)
}

fn perceptron(
  w: [f64; 3],
  x: [f64; 3],
) -> f64 {
  step(dot_product(w, x))
}

fn perceptron_and(x: [f64; 3]) -> f64 {
  perceptron(WEIGHTS_AND, x)
}

fn perceptron_nand(x: [f64; 3]) -> f64 {
  perceptron(WEIGHTS_NAND, x)
}

fn perceptron_or(x: [f64; 3]) -> f64 {
  perceptron(WEIGHTS_OR, x)
}

fn step(x: f64) -> f64 {
  if x.is_sign_positive() {
    1.
  } else {
    0.
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_and() {
    const EXPECTEDS: [f64; 4] = [
      0., 0., 0., 1.,
    ];

    for (input, expected) in INPUTS.iter().zip(EXPECTEDS) {
      let x: [f64; 3] = [
        1., input.0, input.1,
      ];

      let actual = perceptron_and(x);

      assert_eq!(actual, expected);
    }
  }

  #[test]
  fn test_nand() {
    const EXPECTEDS: [f64; 4] = [
      1., 1., 1., 0.,
    ];

    for (input, expected) in INPUTS.iter().zip(EXPECTEDS) {
      let x: [f64; 3] = [
        1., input.0, input.1,
      ];

      let actual = perceptron_nand(x);

      assert_eq!(actual, expected);
    }
  }

  #[test]
  fn test_or() {
    const EXPECTEDS: [f64; 4] = [
      0., 1., 1., 1.,
    ];

    for (input, expected) in INPUTS.iter().zip(EXPECTEDS) {
      let x: [f64; 3] = [
        1., input.0, input.1,
      ];

      let actual = perceptron_or(x);

      assert_eq!(actual, expected);
    }
  }

  #[test]
  fn test_xor() {
    const EXPECTEDS: [f64; 4] = [
      0., 1., 1., 0.,
    ];

    for (input, expected) in INPUTS.iter().zip(EXPECTEDS) {
      let x: [f64; 3] = [
        1., input.0, input.1,
      ];

      let actual = neural_network_xor(x);

      assert_eq!(actual, expected);
    }
  }
}
