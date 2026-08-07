use tch::{Tensor, nn::Linear};

pub struct Mhsa {
  w_q: Linear,
  w_k: Linear,
  w_v: Linear,
  w_o: Linear,
  n_heads: i64,
  d_head: i64,
}

impl Mhsa {
  fn new() -> Self {
    todo!()
  }

  fn forward(&self) -> Tensor {
    todo!()
  }
}
