use super::mhsa::Mhsa;
use ::tch::Tensor;
use ::tch::nn::{LayerNorm, Path, Sequential};

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
    todo!()
  }

  pub fn forward(
    &self,
    x: &Tensor,
    train: bool,
  ) -> Tensor {
    todo!()
  }
}
