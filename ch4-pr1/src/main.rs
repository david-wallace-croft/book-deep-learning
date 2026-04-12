use ::std::time::{Duration, Instant};

const N: u64 = 100_000_000;

fn main() {
  let mut sum: u64 = 0;

  let start: Instant = Instant::now();

  for i in 1..=N {
    sum += i;
  }

  let duration: Duration = start.elapsed();

  println!("sum={sum} duration={duration:?}");
}

#[cfg(test)]
mod test {
  const TEST_N: u64 = 3;

  #[test]
  fn test() {
    let mut sum: u64 = 0;

    for i in 1..=TEST_N {
      sum += i;
    }

    assert_eq!(sum, 6);
  }
}
