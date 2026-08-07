use super::mhsa::Mhsa;
use ::tch::nn::{LayerNorm, Sequential};

pub struct EncoderBlock {
  ln1: LayerNorm,
  ln2: LayerNorm,
  attn: Mhsa,
  ffn: Sequential,
}
