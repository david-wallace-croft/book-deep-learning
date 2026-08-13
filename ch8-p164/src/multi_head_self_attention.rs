use ::tch::nn::{self, Linear, LinearConfig, Path};
use ::tch::{Kind, Tensor};

pub struct MultiHeadSelfAttention {
  query_weighting_layer: Linear,
  key_weighting_layer: Linear,
  value_weighting_layer: Linear,
  output_weighting_layer: Linear,
  heads: i64,
  head_dimensions: i64,
}

impl MultiHeadSelfAttention {
  pub fn new(
    var_stor: &Path,
    model_dimensions: i64,
    heads: i64,
  ) -> Self {
    assert!(
      model_dimensions % heads == 0,
      "model_dimensions must be evenly divisible by heads"
    );

    let head_dimensions: i64 = model_dimensions / heads;

    let linear_config: LinearConfig = LinearConfig {
      bias: true,
      ..Default::default()
    };

    let query_weighting_layer: Linear = nn::linear(
      var_stor / "w_q",
      model_dimensions,
      model_dimensions,
      linear_config,
    );

    let key_weighting_layer: Linear = nn::linear(
      var_stor / "w_k",
      model_dimensions,
      model_dimensions,
      linear_config,
    );

    let value_weighting_layer: Linear = nn::linear(
      var_stor / "w_v",
      model_dimensions,
      model_dimensions,
      linear_config,
    );

    let output_weighting_layer: Linear = nn::linear(
      var_stor / "w_o",
      model_dimensions,
      model_dimensions,
      linear_config,
    );

    Self {
      query_weighting_layer,
      key_weighting_layer,
      value_weighting_layer,
      output_weighting_layer,
      heads,
      head_dimensions,
    }
  }

  pub fn forward(
    &self,
    xs: &Tensor,
    train: bool,
  ) -> Tensor {
    let query_0: Tensor = xs.apply(&self.query_weighting_layer);

    let key_0: Tensor = xs.apply(&self.key_weighting_layer);

    let value_0: Tensor = xs.apply(&self.value_weighting_layer);

    let (batch_size, time_steps, dimensions): (i64, i64, i64) =
      (xs.size()[0], xs.size()[1], xs.size()[2]);

    let split = |x: Tensor| {
      x.view([
        batch_size,
        time_steps,
        self.heads,
        self.head_dimensions,
      ])
      .transpose(1, 2)
    };

    let query_1: Tensor = split(query_0);

    let key_1: Tensor = split(key_0);

    let value_1: Tensor = split(value_0);

    let scale: f64 = (self.head_dimensions as f64).sqrt();

    let scores: Tensor = query_1.matmul(&key_1.transpose(-2, -1)) / scale;

    let attention: Tensor = scores.softmax(-1, Kind::Float);

    // scaled dot-product self-attention
    let context: Tensor = attention.matmul(&value_1);

    let out: Tensor = context.transpose(1, 2).contiguous().view([
      batch_size, time_steps, dimensions,
    ]);

    out.apply_t(&self.output_weighting_layer, train)
  }
}
