use std::slice::Iter;

const NUMBERS: [i32; 5] = [
  1, 2, 3, 4, 5,
];

fn main() {
  let numbers: Vec<i32> = NUMBERS.to_vec();

  let squares: Vec<i32> = square(&numbers);

  println!("Numbers: {numbers:?}");

  println!("Squares: {squares:?}");
}

fn square(numbers: &[i32]) -> Vec<i32> {
  let squares_iter: Iter<i32> = numbers.iter();

  let squares_map = squares_iter.map(|n: &i32| n * n);

  squares_map.collect()
}

#[cfg(test)]
mod test {
  use super::*;

  const SQUARES: [i32; 5] = [
    1, 4, 9, 16, 25,
  ];

  #[test]
  fn test() {
    let numbers: Vec<i32> = NUMBERS.to_vec();

    let actual: Vec<i32> = square(&numbers);

    let expected: Vec<i32> = SQUARES.to_vec();

    assert_eq!(actual, expected);
  }
}
