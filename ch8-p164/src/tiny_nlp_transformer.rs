use super::encoder_block::EncoderBlock;
use ::tch::nn::{
  self, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, Module,
  Path,
};
use ::tch::{Device, Kind, Tensor};

/// A value added to the LayerNorm denominator for numerical stability
const STABILITY_EPSILON: f64 = 1e-5;

pub struct TinyNlpTransformer {
  pub embed: Embedding,
  pub block: EncoderBlock,
  pub layer_norm_f: LayerNorm,
  pub head: Linear,
  pub model_dimensions: i64,
  pub device: Device,
}

impl TinyNlpTransformer {
  pub fn new(
    var_stor: &Path,
    vocabulary_size: i64,
    model_dimensions: i64,
    heads: i64,
    ff_dimensions: i64,
    device: Device,
  ) -> Self {
    let embed: Embedding = nn::embedding(
      var_stor / "embed",
      vocabulary_size,
      model_dimensions,
      EmbeddingConfig::default(),
    );

    let block: EncoderBlock = EncoderBlock::new(
      &(var_stor / "enc0"),
      model_dimensions,
      heads,
      ff_dimensions,
    );

    // https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html
    let layer_norm_f: LayerNorm = nn::layer_norm(
      var_stor / "ln_f",
      vec![model_dimensions],
      LayerNormConfig {
        eps: STABILITY_EPSILON,
        ..Default::default()
      },
    );

    let head: Linear =
      nn::linear(var_stor / "head", model_dimensions, 2, Default::default());

    Self {
      embed,
      block,
      layer_norm_f,
      head,
      model_dimensions,
      device,
    }
  }

  pub fn forward(
    &self,
    x_idx: &Tensor,
    train: bool,
  ) -> Tensor {
    let t: i64 = x_idx.size()[1];

    let pe: Tensor = Self::sinusoidal_positional_encoding(
      t,
      self.model_dimensions,
      self.device,
    );

    let mut x: Tensor = self.embed.forward(x_idx) + pe;

    x = self.block.forward(&x, train);

    let x: Tensor =
      x.apply(&self.layer_norm_f)
        .mean_dim([1].as_slice(), false, Kind::Float);

    x.apply(&self.head)
  }

  fn sinusoidal_positional_encoding(
    time_steps: i64,
    model_dimensions: i64,
    device: Device,
  ) -> Tensor {
    assert!(
      model_dimensions % 2 == 0,
      "Model dimensions must be even for sine/cosine split"
    );

    // Returns a 1-D tensor of size [(end − start) / step]⌉ with values from the
    // interval [start, end) taken with common difference step beginning from
    // start.
    // https://docs.pytorch.org/docs/main/generated/torch.arange.html
    let pos: Tensor =
      Tensor::arange(time_steps, (Kind::Float, device)).unsqueeze(1);

    let i: Tensor = Tensor::arange(model_dimensions / 2, (Kind::Float, device));

    let inv_freq: Tensor =
      ((-10_000.0_f64.ln() * 2. / model_dimensions as f64) as f32 * &i).exp();

    // Returns a new tensor with a dimension of size one inserted at the
    // specified position.
    // https://docs.pytorch.org/docs/main/generated/torch.unsqueeze.html
    let angles: Tensor = &pos * inv_freq.unsqueeze(0);

    // Concatenates the given sequence of tensors in the given dimension.
    // https://docs.pytorch.org/docs/main/generated/torch.cat.html
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
