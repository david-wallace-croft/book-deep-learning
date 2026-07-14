use ::tch::nn::{
  self, Embedding, LayerNorm, LayerNormConfig, Linear, LinearConfig, Path,
  Sequential,
};
use ::tch::{Device, Tensor};
use tch::Kind;

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

  fn combine_heads(
    &self,
    x: &Tensor,
    b: i64,
    t: i64,
  ) -> Tensor {
    x.transpose(1, 2).contiguous().view([
      b,
      t,
      self.n_heads * self.d_head,
    ])
  }

  fn forward_t(
    &self,
    xs: &Tensor,
    train: bool,
  ) -> Tensor {
    let q = xs.apply_t(&self.w_q, train);

    let k = xs.apply_t(&self.w_k, train);

    let v = xs.apply_t(&self.w_v, train);

    let (b, t, _d) = (xs.size()[0], xs.size()[1], xs.size()[2]);

    let q = self.split_heads(&q, b, t);

    let k = self.split_heads(&k, b, t);

    let v = self.split_heads(&v, b, t);

    let scale = (self.d_head as f64).sqrt();

    let scores = q.matmul(&k.transpose(-2, -1)) / scale;

    let mut attn = scores.softmax(-1, Kind::Float);

    if self.dropout_p > 0.0 {
      attn = attn.dropout(self.dropout_p, train);
    }

    let context = attn.matmul(&v);

    let concat = self.combine_heads(&context, b, t);

    concat.apply_t(&self.w_o, train)
  }

  fn split_heads(
    &self,
    x: &Tensor,
    b: i64,
    t: i64,
  ) -> Tensor {
    x.view([
      b,
      t,
      self.n_heads,
      self.d_head,
    ])
    .transpose(1, 2)
  }
}
