use super::encoder_block::EncoderBlock;
use ::tch::nn::{self, Embedding, EmbeddingConfig, LayerNorm, Linear, Path};
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
    let embed: Embedding = nn::embedding(
      var_stor / "embed",
      vocab,
      d_model,
      EmbeddingConfig::default(),
    );

    let mut blocks = Vec::new();

    for i in 0..n_layers {
      let b = EncoderBlock::new(
        &(var_stor / format!("enc{}", i)),
        d_model,
        n_heads,
        d_ff,
        dropout_p,
      );

      blocks.push(b);
    }

    todo!()
  }

  fn forward_t() -> Tensor {
    todo!()
  }
}
