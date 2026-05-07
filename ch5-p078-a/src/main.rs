use ::autodiff::*;

fn main() {
  let f = |x: FT<f64>| x.sin() + x.exp();

  let df = diff(f, 1.);

  println!("df = {df}");
}
