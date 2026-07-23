use super::encoder_block::EncoderBlock;
use ::tch::nn::{
  self, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, Module,
  Path,
};
use ::tch::{Device, Kind, Tensor};

/// A value added to the LayerNorm denominator for numerical stability
const STABILITY_EPSILON: f64 = 1e-5;

pub struct SumModTransformer {
  pub embed: Embedding,
  pub blocks: Vec<EncoderBlock>,
  pub ln_f: LayerNorm,
  pub head: Linear,
  pub model_dimensions: i64,
  #[expect(dead_code)]
  pub time_steps: i64,
  pub dropout_probability: f64,
  pub device: Device,
}

impl SumModTransformer {
  #[expect(clippy::too_many_arguments)]
  pub fn new(
    var_stor: &Path,
    vocabulary_size: i64,
    model_dimensions: i64,
    heads: i64,
    ff_dimensions: i64,
    layers: i64,
    classes: i64,
    time_steps: i64,
    dropout_probability: f64,
    device: Device,
  ) -> Self {
    let embed: Embedding = nn::embedding(
      var_stor / "embed",
      vocabulary_size,
      model_dimensions,
      EmbeddingConfig::default(),
    );

    let mut blocks: Vec<EncoderBlock> = Vec::new();

    for i in 0..layers {
      let b: EncoderBlock = EncoderBlock::new(
        &(var_stor / format!("enc{}", i)),
        model_dimensions,
        heads,
        ff_dimensions,
        dropout_probability,
      );

      blocks.push(b);
    }

    // https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html
    let ln_f: LayerNorm = nn::layer_norm(
      var_stor / "ln_f",
      vec![model_dimensions],
      LayerNormConfig {
        eps: STABILITY_EPSILON,
        ..Default::default()
      },
    );

    let head: Linear = nn::linear(
      var_stor / "head",
      model_dimensions,
      classes,
      Default::default(),
    );

    Self {
      embed,
      blocks,
      ln_f,
      head,
      model_dimensions,
      time_steps,
      dropout_probability,
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
      self.model_dimensions,
      self.device,
    );

    let mut x: Tensor = self.embed.forward(x_idx) + pe;

    if self.dropout_probability > 0. {
      x = x.dropout(self.dropout_probability, train);
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

    // pos is a [[16, 1], Float] with values 0, 1, 2, [...], 15
    // println!("pos: {pos}");

    let i: Tensor = Tensor::arange(model_dimensions / 2, (Kind::Float, device));

    // i is a [[32], Float] with values 0, 1, 2, [...], 31
    // println!("i: {i}");

    let inv_freq: Tensor =
      ((-10_000.0_f64.ln() * 2. / model_dimensions as f64) as f32 * &i).exp();

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
