use super::encoder_block::EncoderBlock;
use ::tch::nn::{
  self, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, Module,
  Path,
};
use ::tch::{Device, Kind, Tensor};

/// A value added to the LayerNorm denominator for numerical stability
const EPSILON: f64 = 1e-5;

pub struct SumModTransformer {
  pub embed: Embedding,
  pub blocks: Vec<EncoderBlock>,
  pub ln_f: LayerNorm,
  pub head: Linear,
  pub d_model: i64,
  #[expect(dead_code)]
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

    let mut blocks: Vec<EncoderBlock> = Vec::new();

    for i in 0..n_layers {
      let b: EncoderBlock = EncoderBlock::new(
        &(var_stor / format!("enc{}", i)),
        d_model,
        n_heads,
        d_ff,
        dropout_p,
      );

      blocks.push(b);
    }

    // https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html
    let ln_f: LayerNorm = nn::layer_norm(
      var_stor / "ln_f",
      vec![d_model],
      LayerNormConfig {
        eps: EPSILON,
        ..Default::default()
      },
    );

    let head: Linear =
      nn::linear(var_stor / "head", d_model, n_classes, Default::default());

    Self {
      embed,
      blocks,
      ln_f,
      head,
      d_model,
      max_t,
      dropout_p,
      device,
    }
  }

  pub fn forward_t(
    &self,
    x_idx: &Tensor,
    train: bool,
  ) -> Tensor {
    let t: i64 = x_idx.size()[1];

    let pe: Tensor = SumModTransformer::sinusoidal_positional_encoding(
      t,
      self.d_model,
      self.device,
    );

    let mut x: Tensor = self.embed.forward(x_idx) + pe;

    if self.dropout_p > 0. {
      x = x.dropout(self.dropout_p, train);
    }

    for b in &self.blocks {
      x = b.forward_t(&x, train);
    }

    let x: Tensor =
      x.apply_t(&self.ln_f, train)
        .mean_dim([1].as_slice(), false, Kind::Float);

    x.apply(&self.head)
  }

  fn sinusoidal_positional_encoding(
    t_steps: i64,
    d_model: i64,
    device: Device,
  ) -> Tensor {
    assert!(
      d_model % 2 == 0,
      "d_model must be even for sine/cosine split"
    );

    let pos: Tensor =
      Tensor::arange(t_steps, (Kind::Float, device)).unsqueeze(1);

    let i: Tensor = Tensor::arange(d_model / 2, (Kind::Float, device));

    let inv_freq: Tensor =
      ((-10_000.0_f64.ln() * 2. / d_model as f64) as f32 * &i).exp();

    let angles: Tensor = &pos * inv_freq.unsqueeze(0);

    Tensor::cat(
      &[
        angles.sin(),
        angles.cos(),
      ],
      1,
    )
    .unsqueeze(0)
  }
}
