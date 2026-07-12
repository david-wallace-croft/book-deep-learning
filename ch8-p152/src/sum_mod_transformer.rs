use super::encoder_block::EncoderBlock;
use ::tch::nn::{Embedding, LayerNorm, Linear, Path};
use ::tch::{Device, Tensor};

pub struct SumModTransformer {
  pub embed: Embedding,
  pub blocks: Vec<EncoderBlock>,
  pub ln_f: LayerNorm,
  pub head: Linear,
  pub d_model: i64,
  pub max_t: i64,
  pub dropout_p: f64,
  pub device: Device,
}

impl SumModTransformer {
  #[expect(clippy::too_many_arguments)]
  pub fn new(
    var_stor: &Path,
    vocab: i64,
    d_model: i64,
    n_heads: i64,
    d_ff: i64,
    n_layers: i64,
    n_classes: i64,
    max_t: i64,
    dropout_p: f64,
    device: Device,
  ) -> Self {
    todo!()
  }

  fn forward_t() -> Tensor {
    todo!()
  }
}
