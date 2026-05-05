use ::autodiff::*;

fn main() {
  let f = |v: &[FT<f64>]| v[0] * v[1].sin() + v[1] * v[1];

  let df = grad(
    f,
    &vec![
      1., 2.,
    ],
  );

  println!("df/dx = {}", df[0]);
  println!("df/dy = {}", df[1]);
}
