use ::tch::nn::{self, Linear, LinearConfig, Path};
use ::tch::{Kind, Tensor};

pub struct MultiHeadSelfAttention {
  w_q: Linear,
  w_k: Linear,
  w_v: Linear,
  w_o: Linear,
  n_heads: i64,
  #[expect(dead_code)]
  d_model: i64,
  d_head: i64,
  dropout_p: f64,
}

impl MultiHeadSelfAttention {
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

    let d_head: i64 = d_model / n_heads;

    let linear_cfg: LinearConfig = LinearConfig {
      bias: true,
      ..Default::default()
    };

    let w_q: Linear =
      nn::linear(var_stor / "w_q", d_model, d_model, linear_cfg);

    let w_k: Linear =
      nn::linear(var_stor / "w_k", d_model, d_model, linear_cfg);

    let w_v: Linear =
      nn::linear(var_stor / "w_v", d_model, d_model, linear_cfg);

    let w_o: Linear =
      nn::linear(var_stor / "w_o", d_model, d_model, linear_cfg);

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
    batch_size: i64,
    time_steps: i64,
  ) -> Tensor {
    // x is [[128, 4, 16, 16], Float]
    // println!("x: {x}");

    // Returns a tensor that is a transposed version of input.
    // The given dimensions dim0 and dim1 are swapped.
    // https://docs.pytorch.org/docs/main/generated/torch.transpose.html
    let x_transpose: Tensor = x.transpose(1, 2);

    // x_transpose is [[128, 16, 4, 16], Float]
    // println!("x_transpose: {x_transpose}");

    // Returns a contiguous in memory tensor containing the same data as self
    // tensor.
    // https://docs.pytorch.org/docs/main/generated/torch.Tensor.contiguous.html
    let x_contiguous: Tensor = x_transpose.contiguous();

    // Returns a new tensor with the same data as the self tensor but of a
    // different shape.
    // https://docs.pytorch.org/docs/main/generated/torch.Tensor.view.html
    x_contiguous.view([
      batch_size,
      time_steps,
      self.n_heads * self.d_head,
    ])
  }

  pub fn forward_t(
    &self,
    xs: &Tensor,
    train: bool,
  ) -> Tensor {
    let q: Tensor = xs.apply_t(&self.w_q, train);

    let k: Tensor = xs.apply_t(&self.w_k, train);

    let v: Tensor = xs.apply_t(&self.w_v, train);

    let (b, t, _d): (i64, i64, i64) =
      (xs.size()[0], xs.size()[1], xs.size()[2]);

    let q: Tensor = self.split_heads(&q, b, t);

    let k: Tensor = self.split_heads(&k, b, t);

    let v: Tensor = self.split_heads(&v, b, t);

    let scale: f64 = (self.d_head as f64).sqrt();

    let scores: Tensor = q.matmul(&k.transpose(-2, -1)) / scale;

    let mut attn: Tensor = scores.softmax(-1, Kind::Float);

    if self.dropout_p > 0.0 {
      attn = attn.dropout(self.dropout_p, train);
    }

    let context: Tensor = attn.matmul(&v);

    let concat: Tensor = self.combine_heads(&context, b, t);

    concat.apply_t(&self.w_o, train)
  }

  fn split_heads(
    &self,
    x: &Tensor,
    batch_size: i64,
    time_steps: i64,
  ) -> Tensor {
    // x is [[128, 16, 64], Float]
    // println!("x: {x}");

    let x_view: Tensor = x.view([
      batch_size,
      time_steps,
      self.n_heads,
      self.d_head,
    ]);

    // x_view is [[128, 16, 4, 16], Float]
    // println!("x_view: {x_view}");

    let x_transpose: Tensor = x_view.transpose(1, 2);

    // x_transpose is [[128, 4, 16, 16], Float]
    // println!("x_transpose: {x_transpose}");

    x_transpose
  }
}
