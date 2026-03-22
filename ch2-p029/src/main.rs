use ::autograd::Graph;
use ::autograd::Tensor;
use ::autograd::ndarray;
use ::autograd::tensor::Variable;

fn main() {
  ::autograd::with(|graph: &mut Graph<f64>| {
    let x: Tensor<'_, f64> = graph.variable(ndarray::arr0(3.));

    let y: Tensor<'_, f64> = x * x;

    let grads: Vec<Tensor<'_, f64>> = graph.grad(&[y], &[x]);

    match grads[0].eval(&[]) {
      Ok(grad) => println!("Gradient of y with respect to x: {grad:?}"),
      Err(e) => println!("Error computing gradient: {e:?}"),
    }
  });
}

#[cfg(test)]
mod test {
  use super::*;
  use ::autograd::EvalError;
  use ::autograd::ndarray::ArrayBase;
  use ::autograd::ndarray::Dim;
  use ::autograd::ndarray::IxDynImpl;
  use ::autograd::ndarray::OwnedRepr;

  #[test]
  fn test() {
    ::autograd::with(|graph: &mut Graph<f64>| {
      let x: Tensor<'_, f64> = graph.variable(ndarray::arr0(3.));

      let y: Tensor<'_, f64> = x * x;

      let grads: Vec<Tensor<'_, f64>> = graph.grad(&[y], &[x]);

      let result: Result<ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>, EvalError> =
        grads[0].eval(&[]);

      let array_base: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>> =
        result.unwrap();

      let actual: &String = &format!("{array_base:?}");

      assert_eq!(
        actual,
        "6.0 shape=[], strides=[], layout=C | F (0x3), dynamic ndim=0"
      );
    });
  }
}
