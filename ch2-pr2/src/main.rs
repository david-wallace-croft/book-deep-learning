use ::ndarray::Array;

fn main() {
  let arr = Array::from_vec(vec![
    5, 10, 15,
  ]);

  println!("Array: {arr:?}");
}
