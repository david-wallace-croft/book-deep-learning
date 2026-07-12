use ::tch::Device;
use ::tch::nn::{Embedding, LayerNorm, Linear, Sequential};

pub struct EncoderBlock {
  pub embed: Embedding,
  pub blocks: Vec<EncoderBlock>,
  pub ln_f: LayerNorm,
  pub head: Linear,
  pub d_model: i64,
  pub max_t: i64,
  pub dropout_p: f64,
  pub device: Device,
}
