const INPUTS: [(u8, u8); 4] = [
  (0, 0),
  (0, 1),
  (1, 0),
  (1, 1),
];

fn main() {
  for (x, y) in INPUTS {
    let output = perceptron(x, y);

    println!("Input: ({}, {}) => AND: {}", x, y, output);
  }
}

fn perceptron(
  x: u8,
  y: u8,
) -> u8 {
  let w1 = 1.;

  let w2 = 1.;

  let bias = -1.5;

  let sum = (x as f64) * w1 + (y as f64) * w2 + bias;

  step(sum)
}

fn step(x: f64) -> u8 {
  if x >= 0. {
    1
  } else {
    0
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
    const EXPECTEDS: [u8; 4] = [
      0, 0, 0, 1,
    ];

    for (input, expected) in INPUTS.iter().zip(EXPECTEDS) {
      let actual = perceptron(input.0, input.1);

      assert_eq!(actual, expected);
    }
  }
}
