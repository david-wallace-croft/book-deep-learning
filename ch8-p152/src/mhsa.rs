use ::tch::nn::{
  self, Embedding, LayerNorm, LayerNormConfig, Linear, LinearConfig, Path,
  Sequential,
};
use ::tch::{Device, Tensor};

#[expect(clippy::upper_case_acronyms)]
pub struct MHSA {
  w_q: Linear,
  w_k: Linear,
  w_v: Linear,
  w_o: Linear,
  n_heads: i64,
  d_model: i64,
  d_head: i64,
  dropout_p: f64,
}

impl MHSA {
  pub fn new(
    var_stor: &Path,
    d_model: i64,
    n_heads: i64,
    dropout_p: f64,
  ) -> Self {
    assert!(
      d_model % n_heads == 0,
      "d_model must be divisible by n_heads"
    );

    let d_head = d_model / n_heads;

    let linear_cfg = LinearConfig {
      bias: true,
      ..Default::default()
    };

    let w_q = nn::linear(var_stor / "w_q", d_model, d_model, linear_cfg);

    let w_k = nn::linear(var_stor / "w_k", d_model, d_model, linear_cfg);

    let w_v = nn::linear(var_stor / "w_v", d_model, d_model, linear_cfg);

    let w_o = nn::linear(var_stor / "w_o", d_model, d_model, linear_cfg);

    Self {
      w_q,
      w_k,
      w_v,
      w_o,
      n_heads,
      d_model,
      d_head,
      dropout_p,
    }
  }

  fn forward_t() -> Tensor {
    todo!()
  }

  fn split_heads() -> Tensor {
    todo!()
  }

  fn combine_heads() -> Tensor {
    todo!()
  }
}
