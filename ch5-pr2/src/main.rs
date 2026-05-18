use ::autodiff::*;

fn main() {
  let f = |v: &[FT<f64>]| -> FT<f64> { (v[0] - v[1]) * (v[0] - v[1]) };

  let v: Vec<F<f64, f64>> = vec![
    F1::cst(3.),
    F1::var(2.),
  ];

  let dual: F<f64, f64> = f(&v);

  println!("{dual}");
}
