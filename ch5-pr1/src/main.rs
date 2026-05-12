use ::autodiff::*;

fn main() {
  let x: FT<f64> = FT {
    x: 2.5,
    dx: 1.,
  };

  let f = |x: FT<f64>| -> FT<f64> { 3. * x * x + 2. * x + 1. };

  let dual: FT<f64> = f(x);

  println!("dual: {dual}");
}
