use ::autograd::tensor::Variable;

fn main() {
  ::autograd::with(|graph| {
    let x = graph.variable(::autograd::ndarray::arr0(4.));

    let y = x * x * x;

    let grads = graph.grad(&[y], &[x]);

    match grads[0].eval(&[]) {
      Ok(grad) => println!("Gradient of y with respect to x: {grad:?}"),
      Err(e) => println!("Error computing gradient: {e:?}"),
    }
  });
}
