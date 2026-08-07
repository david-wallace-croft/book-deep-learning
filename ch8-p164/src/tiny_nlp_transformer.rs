use super::encoder_block::EncoderBlock;
use ::tch::nn::{Embedding, LayerNorm, Linear, Path};
use ::tch::{Device, Tensor};

pub struct TinyNlpTransformer {
  pub embed: Embedding,
  pub block: EncoderBlock,
  pub ln_f: LayerNorm,
  pub head: Linear,
  pub d_model: i64,
  pub device: Device,
}

impl TinyNlpTransformer {
  pub fn new(
    vs: &Path,
    vocab: i64,
    d_model: i64,
    n_heads: i64,
    d_ff: i64,
    device: Device,
  ) -> Self {
    todo!()
  }

  pub fn forward(
    &self,
    x_idx: &Tensor,
    train: bool,
  ) -> Tensor {
    todo!()
  }
}
