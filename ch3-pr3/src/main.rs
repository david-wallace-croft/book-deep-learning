const INPUTS: [i32; 3] = [
  -7, 0, 42,
];

fn main() {
  for input in INPUTS {
    let description = describe_number(input);

    println!("{input} is {description}");
  }
}

fn describe_number(n: i32) -> &'static str {
  match n {
    0 => "zero",
    i32::MIN..=-1_i32 => "negative",
    1_i32..=i32::MAX => "positive",
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
    const EXPECTEDS: [&'static str; 3] = [
      "negative", "zero", "positive",
    ];

    for (index, input) in INPUTS.into_iter().enumerate() {
      let actual = describe_number(input);

      let expected = EXPECTEDS[index];

      assert_eq!(actual, expected);
    }
  }
}
