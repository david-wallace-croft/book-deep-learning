use super::multi_head_self_attention::MultiHeadSelfAttention;
use ::tch::Tensor;
use ::tch::nn::{
  self, LayerNorm, LayerNormConfig, LinearConfig, Path, Sequential,
};

const STABILITY_EPSILON: f64 = 1e-5;

pub struct EncoderBlock {
  feed_forward_network: Sequential,
  layer_norm_1: LayerNorm,
  layer_norm_2: LayerNorm,
  multi_head_self_attention: MultiHeadSelfAttention,
}

impl EncoderBlock {
  pub fn new(
    var_stor: &Path,
    model_dimensions: i64,
    heads: i64,
    feed_forward_dimensions: i64,
  ) -> Self {
    let layer_norm_config: LayerNormConfig = LayerNormConfig {
      eps: STABILITY_EPSILON,
      ..Default::default()
    };

    let layer_norm_1: LayerNorm = nn::layer_norm(
      var_stor / "ln1",
      vec![model_dimensions],
      layer_norm_config,
    );

    let layer_norm_2: LayerNorm = nn::layer_norm(
      var_stor / "ln2",
      vec![model_dimensions],
      layer_norm_config,
    );

    let multi_head_self_attention: MultiHeadSelfAttention =
      MultiHeadSelfAttention::new(
        &(var_stor / "attn"),
        model_dimensions,
        heads,
      );

    let feed_forward_network: Sequential = nn::seq()
      .add(nn::linear(
        var_stor / "ff1",
        model_dimensions,
        feed_forward_dimensions,
        LinearConfig::default(),
      ))
      // Applies the Gaussian Error Linear Units function
      // https://docs.pytorch.org/docs/main/generated/torch.nn.GELU.html
      .add_fn(|x: &Tensor| x.gelu("tanh"))
      .add(nn::linear(
        var_stor / "ff2",
        feed_forward_dimensions,
        model_dimensions,
        LinearConfig::default(),
      ));

    Self {
      layer_norm_1,
      layer_norm_2,
      multi_head_self_attention,
      feed_forward_network,
    }
  }

  pub fn forward(
    &self,
    input_x: &Tensor,
    train: bool,
  ) -> Tensor {
    // Applies Layer Normalization over a mini-batch of inputs.
    // https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html
    let normalized: Tensor = input_x.apply(&self.layer_norm_1);
    let attention_output: Tensor =
      self.multi_head_self_attention.forward(&normalized, train);

    let residual_connection_1: Tensor = input_x + attention_output;

    let h2: Tensor = residual_connection_1
      .apply(&self.layer_norm_2)
      .apply_t(&self.feed_forward_network, train);

    let residual_connection_2 = residual_connection_1 + h2;

    // No clippy "unnecessary 'let' binding" warning if you put a comment here

    residual_connection_2
  }
}
