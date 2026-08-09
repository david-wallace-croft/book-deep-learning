use ::tch::nn::{self, Linear, LinearConfig, Path};
use ::tch::{Kind, Tensor};

pub struct Mhsa {
  w_q: Linear,
  w_k: Linear,
  w_v: Linear,
  w_o: Linear,
  n_heads: i64,
  d_head: i64,
}

impl Mhsa {
  fn new(
    vs: &Path,
    d_model: i64,
    n_heads: i64,
  ) -> Self {
    assert!(
      d_model % n_heads == 0,
      "d_model must be evenly divisible by n_heads"
    );

    let cfg = LinearConfig {
      bias: true,
      ..Default::default()
    };

    let w_q = nn::linear(vs / "w_q", d_model, d_model, cfg);

    let w_k = nn::linear(vs / "w_k", d_model, d_model, cfg);

    let w_v = nn::linear(vs / "w_v", d_model, d_model, cfg);

    let w_o = nn::linear(vs / "w_o", d_model, d_model, cfg);

    Self {
      w_q,
      w_k,
      w_v,
      w_o,
      n_heads,
      d_head: d_model / n_heads,
    }
  }

  fn forward(
    &self,
    xs: &Tensor,
    train: bool,
  ) -> Tensor {
    let (b, t, d) = (xs.size()[0], xs.size()[1], xs.size()[2]);

    let q = xs.apply(&self.w_q);

    let k = xs.apply(&self.w_k);

    let v = xs.apply(&self.w_v);

    let split = |x: Tensor| {
      x.view([
        b,
        t,
        self.n_heads,
        self.d_head,
      ])
      .transpose(1, 2)
    };

    let q = split(q);

    let k = split(k);

    let v = split(v);

    let scale = (self.d_head as f64).sqrt();

    let scores = q.matmul(&k.transpose(-2, -1)) / scale;

    let attn = scores.softmax(-1, Kind::Float);

    let ctx = attn.matmul(&v);

    let out = ctx.transpose(1, 2).contiguous().view([
      b, t, d,
    ]);

    out.apply_t(&self.w_o, train)
  }
}
