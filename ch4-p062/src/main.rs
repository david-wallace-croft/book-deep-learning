use ::rayon::prelude::*;
use ::std::time::{Duration, Instant};

// Running in development mode exits with an overflow error for N > 1_000_000:
// cargo run -p ch4-p062 -q
// Running in release mode shows a speedup of less than one for small N:
// cargo run -p ch4-p062 -q --release
const N: u64 = 1_000_000;

fn main() {
  let data: Vec<u64> = (0..N).collect();

  let start: Instant = Instant::now();

  let sum_serial: u64 = sum_serial(&data);

  let duration_serial: Duration = start.elapsed();

  let start: Instant = Instant::now();

  let sum_parallel: u64 = sum_parallel(&data);

  let duration_parallel: Duration = start.elapsed();

  let speedup: f64 =
    duration_serial.as_secs_f64() / duration_parallel.as_secs_f64();

  println!("Serial .....: sum={sum_serial} duration={duration_serial:?}");

  println!("Parallel ...: sum={sum_parallel} duration={duration_parallel:?}");

  println!("Speedup ....: {speedup}");
}

fn square(x: u64) -> u64 {
  x * x
}

fn sum_parallel(data: &[u64]) -> u64 {
  // The sum can overflow
  data.par_iter().map(|&x| square(x)).sum()
}

fn sum_serial(data: &[u64]) -> u64 {
  // The sum can overflow
  data.iter().map(|&x| square(x)).sum()
}

#[cfg(test)]
mod test {
  use super::*;

  const TEST_N: u64 = 1_000;

  #[test]
  fn test() {
    let data: Vec<u64> = (0..TEST_N).collect();

    let sum_serial: u64 = sum_serial(&data);

    let sum_parallel: u64 = sum_parallel(&data);

    assert_eq!(sum_parallel, sum_serial);
  }
}
