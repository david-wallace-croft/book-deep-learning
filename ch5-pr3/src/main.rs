use ::autodiff::*;

fn main() {
  const W: [f64; 3] = [
    0.1, 0.3, 0.6,
  ];

  const X: [f64; 3] = [
    1., 2., -1.,
  ];

  let w: Vec<FT<f64>> = W.iter().map(|w| FT::cst(*w)).collect();

  let x: Vec<FT<f64>> = X.iter().map(|x| FT::cst(*x)).collect();

  let f = |x: &[FT<f64>]| -> FT<f64> {
    let z: FT<f64> = w
      .iter()
      .zip(x.iter())
      .map(|(w, x): (&FT<f64>, &FT<f64>)| -> FT<f64> { *w * *x })
      .sum();

    relu(z)
  };

  let y: FT<f64> = f(&x);

  let g: Vec<f64> = grad(f, &X);

  println!("(y, dy) = {}", y);

  println!("y = {:.1}", y.value());

  println!("\u{2207}\u{2093}y = {:?}", g);

  println!("\u{2202}y/\u{2202}x\u{2081} = {}", g[1]);
}

fn relu(x: FT<f64>) -> FT<f64> {
  if x.x > 0. {
    x
  } else {
    FT::cst(0.)
  }
}
