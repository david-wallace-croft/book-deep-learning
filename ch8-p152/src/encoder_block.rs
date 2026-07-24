use super::multi_head_self_attention::MultiHeadSelfAttention;
use ::tch::Tensor;
use ::tch::nn::{
  self, LayerNorm, LayerNormConfig, LinearConfig, Path, Sequential,
};

const STABILITY_EPSILON: f64 = 1e-5;

pub struct EncoderBlock {
  pub dropout_probability: f64,
  pub feed_forward_network: Sequential,
  pub layer_norm_1: LayerNorm,
  pub layer_norm_2: LayerNorm,
  pub multi_head_self_attention: MultiHeadSelfAttention,
}

impl EncoderBlock {
  pub fn new(
    var_stor: &Path,
    model_dimensions: i64,
    heads: i64,
    ff_dimensions: i64,
    dropout_probability: f64,
  ) -> Self {
    let ln_cfg: LayerNormConfig = LayerNormConfig {
      eps: STABILITY_EPSILON,
      ..Default::default()
    };

    let layer_norm_1: LayerNorm =
      nn::layer_norm(var_stor / "ln1", vec![model_dimensions], ln_cfg);

    let layer_norm_2: LayerNorm =
      nn::layer_norm(var_stor / "ln2", vec![model_dimensions], ln_cfg);

    let multi_head_self_attention: MultiHeadSelfAttention =
      MultiHeadSelfAttention::new(
        &(var_stor / "attn"),
        model_dimensions,
        heads,
        dropout_probability,
      );

    let feed_forward_network: Sequential = nn::seq()
      .add(nn::linear(
        var_stor / "ff1",
        model_dimensions,
        ff_dimensions,
        LinearConfig::default(),
      ))
      .add_fn(|x: &Tensor| x.gelu("tanh"))
      .add(nn::linear(
        var_stor / "ff2",
        ff_dimensions,
        model_dimensions,
        LinearConfig::default(),
      ));

    Self {
      layer_norm_1,
      layer_norm_2,
      multi_head_self_attention,
      feed_forward_network,
      dropout_probability,
    }
  }

  pub fn forward_t(
    &self,
    x: &Tensor,
    train: bool,
  ) -> Tensor {
    // Applies Layer Normalization over a mini-batch of inputs.
    // https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html
    // Applies the function callable to each element in the tensor, replacing
    // each element with the value returned by callable.
    // https://docs.pytorch.org/docs/main/generated/torch.Tensor.apply_.html
    // apply_t() might be an implementation of the Visitor pattern.
    let h: Tensor = x.apply_t(&self.layer_norm_1, train);

    let mut h: Tensor = self.multi_head_self_attention.forward_t(&h, train);

    if self.dropout_probability > 0. {
      h = h.dropout(self.dropout_probability, train);
    }

    let x: Tensor = x + h;

    let h2: Tensor = x
      .apply_t(&self.layer_norm_2, train)
      .apply_t(&self.feed_forward_network, train);

    let h2: Tensor = if self.dropout_probability > 0. {
      // During training, randomly zeroes some of the elements of the input
      // tensor with probability p.
      // https://docs.pytorch.org/docs/main/generated/torch.nn.Dropout.html
      h2.dropout(self.dropout_probability, train)
    } else {
      h2
    };

    x + h2
  }
}
