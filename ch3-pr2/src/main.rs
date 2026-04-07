fn main() {
  let mut counter: i32 = 0;

  increment(&mut counter);

  // increment(&counter);
}

fn increment(x: &mut i32) {
  *x += 1;
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() {
    let mut counter: i32 = 0;

    increment(&mut counter);

    assert_eq!(counter, 1);
  }
}
