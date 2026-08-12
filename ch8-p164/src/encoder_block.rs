use super::mhsa::Mhsa;
use ::tch::Tensor;
use ::tch::nn::{self, LayerNorm, LayerNormConfig, Path, Sequential};

pub struct EncoderBlock {
  ln1: LayerNorm,
  ln2: LayerNorm,
  attn: Mhsa,
  ffn: Sequential,
}

impl EncoderBlock {
  pub fn new(
    vs: &Path,
    d_model: i64,
    n_heads: i64,
    d_ff: i64,
  ) -> Self {
    let ln_cfg = LayerNormConfig {
      eps: 1e-5,
      ..Default::default()
    };

    let ln1 = nn::layer_norm(vs / "ln1", vec![d_model], ln_cfg);

    let ln2 = nn::layer_norm(vs / "ln2", vec![d_model], ln_cfg);

    let attn = Mhsa::new(&(vs / "attn"), d_model, n_heads);

    let ffn = nn::seq()
      .add(nn::linear(vs / "ff1", d_model, d_ff, Default::default()))
      .add_fn(|x| x.gelu("tanh"))
      .add(nn::linear(vs / "ff2", d_ff, d_model, Default::default()));

    Self {
      ln1,
      ln2,
      attn,
      ffn,
    }
  }

  pub fn forward(
    &self,
    x: &Tensor,
    train: bool,
  ) -> Tensor {
    let h = self.attn.forward(&x.apply(&self.ln1), train);

    let x = x + h;

    let h2 = x.apply(&self.ln2).apply_t(&self.ffn, train);

    x + h2
  }
}
