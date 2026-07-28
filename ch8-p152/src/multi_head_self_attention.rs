use ::tch::nn::{self, Linear, LinearConfig, Path};
use ::tch::{Kind, Tensor};

pub struct MultiHeadSelfAttention {
  query_weighting_layer: Linear,
  key_weighting_layer: Linear,
  value_weighting_layer: Linear,
  output_weighting_layer: Linear,
  heads: i64,
  #[expect(dead_code)]
  model_dimensions: i64,
  head_dimensions: i64,
  dropout_probability: f64,
}

impl MultiHeadSelfAttention {
  pub fn new(
    var_stor: &Path,
    model_dimensions: i64,
    heads: i64,
    dropout_probability: f64,
  ) -> Self {
    assert!(
      model_dimensions % heads == 0,
      "model_dimensions must be evenly divisible by heads"
    );

    let head_dimensions: i64 = model_dimensions / heads;

    let linear_cfg: LinearConfig = LinearConfig {
      bias: true,
      ..Default::default()
    };

    let query_weighting_layer: Linear = nn::linear(
      var_stor / "w_q",
      model_dimensions,
      model_dimensions,
      linear_cfg,
    );

    let key_weighting_layer: Linear = nn::linear(
      var_stor / "w_k",
      model_dimensions,
      model_dimensions,
      linear_cfg,
    );

    let value_weighting_layer: Linear = nn::linear(
      var_stor / "w_v",
      model_dimensions,
      model_dimensions,
      linear_cfg,
    );

    let output_weighting_layer: Linear = nn::linear(
      var_stor / "w_o",
      model_dimensions,
      model_dimensions,
      linear_cfg,
    );

    Self {
      query_weighting_layer,
      key_weighting_layer,
      value_weighting_layer,
      output_weighting_layer,
      heads,
      model_dimensions,
      head_dimensions,
      dropout_probability,
    }
  }

  fn combine_heads(
    &self,
    x: &Tensor,
    batch_size: i64,
    time_steps: i64,
  ) -> Tensor {
    // x is [[128, 4, 16, 16], Float]
    // println!("x: {x}");

    // Returns a tensor that is a transposed version of input.
    // The given dimensions dim0 and dim1 are swapped.
    // https://docs.pytorch.org/docs/main/generated/torch.transpose.html
    let x_transpose: Tensor = x.transpose(1, 2);

    // x_transpose is [[128, 16, 4, 16], Float]
    // println!("x_transpose: {x_transpose}");

    // Returns a contiguous in memory tensor containing the same data as self
    // tensor.
    // https://docs.pytorch.org/docs/main/generated/torch.Tensor.contiguous.html
    let x_contiguous: Tensor = x_transpose.contiguous();

    // Returns a new tensor with the same data as the self tensor but of a
    // different shape.
    // https://docs.pytorch.org/docs/main/generated/torch.Tensor.view.html
    x_contiguous.view([
      batch_size,
      time_steps,
      self.heads * self.head_dimensions,
    ])
  }

  pub fn forward_t(
    &self,
    xs: &Tensor,
    train: bool,
  ) -> Tensor {
    let query_0: Tensor = xs.apply_t(&self.query_weighting_layer, train);

    let key_0: Tensor = xs.apply_t(&self.key_weighting_layer, train);

    let value_0: Tensor = xs.apply_t(&self.value_weighting_layer, train);

    let (batch_size, time_steps, _dimensions): (i64, i64, i64) =
      (xs.size()[0], xs.size()[1], xs.size()[2]);

    let query_1: Tensor = self.split_heads(&query_0, batch_size, time_steps);

    let key_1: Tensor = self.split_heads(&key_0, batch_size, time_steps);

    let value_1: Tensor = self.split_heads(&value_0, batch_size, time_steps);

    let scale: f64 = (self.head_dimensions as f64).sqrt();

    let scores: Tensor = query_1.matmul(&key_1.transpose(-2, -1)) / scale;

    let mut attention: Tensor = scores.softmax(-1, Kind::Float);

    if self.dropout_probability > 0.0 {
      attention = attention.dropout(self.dropout_probability, train);
    }

    // scaled dot-product self-attention
    let context: Tensor = attention.matmul(&value_1);

    let concatenation: Tensor =
      self.combine_heads(&context, batch_size, time_steps);

    concatenation.apply_t(&self.output_weighting_layer, train)
  }

  fn split_heads(
    &self,
    x: &Tensor,
    batch_size: i64,
    time_steps: i64,
  ) -> Tensor {
    // x is [[128, 16, 64], Float]
    // println!("x: {x}");

    let x_view: Tensor = x.view([
      batch_size,
      time_steps,
      self.heads,
      self.head_dimensions,
    ]);

    // x_view is [[128, 16, 4, 16], Float]
    // println!("x_view: {x_view}");

    let x_transpose: Tensor = x_view.transpose(1, 2);

    // x_transpose is [[128, 4, 16, 16], Float]
    // println!("x_transpose: {x_transpose}");

    x_transpose
  }
}
