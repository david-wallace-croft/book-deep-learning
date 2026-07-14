use super::mhsa::MHSA;
use ::tch::nn::{
  self, Embedding, LayerNorm, LayerNormConfig, Linear, Path, Sequential,
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
    let ln_cfg = LayerNormConfig {
      eps: 1e-5,
      ..Default::default()
    };

    let ln1 = nn::layer_norm(var_stor / "ln1", vec![d_model], ln_cfg);

    let ln2 = nn::layer_norm(var_stor / "ln2", vec![d_model], ln_cfg);

    let attn = MHSA::new(&(var_stor / "attn"), d_model, n_heads, dropout_p);

    todo!()
  }

  fn forward_t() -> Tensor {
    todo!()
  }
}
