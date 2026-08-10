use super::encoder_block::EncoderBlock;
use ::tch::nn::{self, Embedding, LayerNorm, Linear, Module, Path};
use ::tch::{Device, Kind, Tensor};

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
    let embed = nn::embedding(vs / "embed", vocab, d_model, Default::default());

    let block = EncoderBlock::new(&(vs / "enc0"), d_model, n_heads, d_ff);

    let ln_f = nn::layer_norm(
      vs / "ln_f",
      vec![d_model],
      nn::LayerNormConfig {
        eps: 1e-5,
        ..Default::default()
      },
    );

    let head = nn::linear(vs / "head", d_model, 2, Default::default());

    Self {
      embed,
      block,
      ln_f,
      head,
      d_model,
      device,
    }
  }

  pub fn forward(
    &self,
    x_idx: &Tensor,
    train: bool,
  ) -> Tensor {
    let t = x_idx.size()[1];

    let pe = Self::sinusoidal_pe(t, self.d_model, self.device);

    let mut x = self.embed.forward(x_idx) + pe;

    x = self.block.forward(&x, train);

    let x = x
      .apply(&self.ln_f)
      .mean_dim([1].as_slice(), false, Kind::Float);

    x.apply(&self.head)
  }

  fn sinusoidal_pe(
    t_steps: i64,
    d_model: i64,
    device: Device,
  ) -> Tensor {
    todo!()
  }
}
