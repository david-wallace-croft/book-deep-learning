use super::mhsa::MHSA;
use ::tch::nn::{
  self, Embedding, LayerNorm, LayerNormConfig, Linear, LinearConfig, Path,
  Sequential,
};
use ::tch::{Device, Tensor};

pub struct EncoderBlock {
  pub ln1: LayerNorm,
  pub ln2: LayerNorm,
  pub attn: MHSA,
  pub ffn: Sequential,
  pub dropout_p: f64,
}

impl EncoderBlock {
  pub fn new(
    var_stor: &Path,
    d_model: i64,
    n_heads: i64,
    d_ff: i64,
    dropout_p: f64,
  ) -> Self {
    let ln_cfg: LayerNormConfig = LayerNormConfig {
      eps: 1e-5,
      ..Default::default()
    };

    let ln1: LayerNorm =
      nn::layer_norm(var_stor / "ln1", vec![d_model], ln_cfg);

    let ln2: LayerNorm =
      nn::layer_norm(var_stor / "ln2", vec![d_model], ln_cfg);

    let attn: MHSA =
      MHSA::new(&(var_stor / "attn"), d_model, n_heads, dropout_p);

    let ffn: Sequential = nn::seq()
      .add(nn::linear(
        var_stor / "ff1",
        d_model,
        d_ff,
        LinearConfig::default(),
      ))
      .add_fn(|x: &Tensor| x.gelu("tanh"))
      .add(nn::linear(
        var_stor / "ff2",
        d_ff,
        d_model,
        LinearConfig::default(),
      ));

    Self {
      ln1,
      ln2,
      attn,
      ffn,
      dropout_p,
    }
  }

  fn forward_t(
    &self,
    x: &Tensor,
    train: bool,
  ) -> Tensor {
    let h = x.apply_t(&self.ln1, train);

    let mut h = self.attn.forward_t(&h, train);

    if self.dropout_p > 0. {
      h = h.dropout(self.dropout_p, train);
    }

    let x = x + h;

    let h2 = x.apply_t(&self.ln2, train).apply_t(&self.ffn, train);

    let h2 = if self.dropout_p > 0. {
      h2.dropout(self.dropout_p, train)
    } else {
      h2
    };

    x + h2
  }
}
