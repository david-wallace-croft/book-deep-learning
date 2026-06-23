use super::aliases::{Dataset, Image, Matrix, Vector};

pub struct Network {
  pub conv_out: Matrix,
  pub fc_bias: f32,
  pub fc_weight_matrix: Matrix,
  pub flat: Vector,
  pub kernel: Matrix,
  pub label_count: usize,
  pub learning_rate: f32,
  pub max_pos: Vec<Vec<(usize, usize)>>,
  pub pool_out: Matrix,
  pub total_loss: f32,
}

impl Network {
  pub fn backprop(
    &mut self,
    image: &Image,
    weight_index: usize,
    y_pred: f32,
    y_true: f32,
  ) {
    let loss: f32 = super::binary_cross_entropy(y_true, y_pred);

    self.total_loss += loss;

    let dz: f32 = y_pred - y_true;

    let fc_weights = &mut self.fc_weight_matrix[weight_index];

    for i in 0..fc_weights.len() {
      fc_weights[i] -= self.learning_rate * dz * self.flat[i];
    }

    self.fc_bias -= self.learning_rate * dz;

    let mut d_pool_out: Matrix =
      vec![vec![0.; self.pool_out[0].len()]; self.pool_out.len()];

    let mut idx: usize = 0;

    #[expect(clippy::needless_range_loop)]
    for i in 0..self.pool_out.len() {
      for j in 0..self.pool_out[0].len() {
        d_pool_out[i][j] = dz * fc_weights[idx];

        idx += 1;
      }
    }

    let d_conv_out: Matrix = super::max_pool2x2_backprop(
      &d_pool_out,
      &self.max_pos,
      self.conv_out.len(),
      self.conv_out[0].len(),
    );

    let mut d_conv_out_relu: Matrix =
      vec![vec![0.; self.conv_out[0].len()]; self.conv_out.len()];

    for i in 0..self.conv_out.len() {
      for j in 0..self.conv_out[0].len() {
        d_conv_out_relu[i][j] =
          d_conv_out[i][j] * super::relu_deriv(self.conv_out[i][j]);
      }
    }

    super::conv2d_backprop(
      &d_conv_out_relu,
      image,
      &mut self.kernel,
      self.learning_rate,
    );
  }

  pub fn calc_flat_length(
    kernel: &Matrix,
    train_dataset: &Dataset,
  ) -> usize {
    let temp_conv: Matrix = super::conv2d(&train_dataset[0].0, &kernel);

    let (temp_pool, _): (Matrix, Vec<Vec<(usize, usize)>>) =
      super::max_pool2x2(&temp_conv);

    temp_pool.len() * temp_pool[0].len()
  }

  pub fn train(
    &mut self,
    train_dataset: &Dataset,
  ) {
    self.total_loss = 0.;

    for (index, (image, label)) in train_dataset.iter().enumerate() {
      if index % 10_000 == 0 {
        println!("index = {index}");
      }

      self.conv_out = super::conv2d(image, &self.kernel);

      for i in 0..self.conv_out.len() {
        for j in 0..self.conv_out[0].len() {
          self.conv_out[i][j] = super::relu(self.conv_out[i][j]);
        }
      }

      let (pool_out, max_pos): (Matrix, Vec<Vec<(usize, usize)>>) =
        super::max_pool2x2(&self.conv_out);

      self.pool_out = pool_out;

      self.max_pos = max_pos;

      self.flat = super::flatten(&self.pool_out);

      for weight_index in 0..self.label_count {
        let fc_weights = &self.fc_weight_matrix[weight_index];

        let y_pred = super::perceptron(&self.flat, fc_weights, self.fc_bias);

        let y_true = if *label as usize == weight_index {
          1.
        } else {
          0.
        };

        self.backprop(image, weight_index, y_pred, y_true);
      }
    }
  }

  pub fn new(train_dataset: &Dataset) -> Network {
    let kernel: Matrix = vec![
      vec![
        0.1, 0.2, -0.1,
      ],
      vec![
        0.0, 0.0, 0.1,
      ],
      vec![
        -0.2, 0., 0.2,
      ],
    ];

    let flat_length: usize = Self::calc_flat_length(&kernel, train_dataset);

    let label_count: usize = 10;

    let fc_weight_matrix: Matrix = vec![vec![0.5; flat_length]; label_count];

    Network {
      conv_out: Default::default(),
      fc_bias: 0.,
      fc_weight_matrix,
      flat: Default::default(),
      kernel,
      label_count,
      learning_rate: 0.001,
      max_pos: Default::default(),
      pool_out: Default::default(),
      total_loss: 0.,
    }
  }

  pub fn predict(
    &self,
    image: &[Vector],
    weight_index: usize,
  ) -> f32 {
    let mut conv_out: Matrix = super::conv2d(image, &self.kernel);

    for row in conv_out.iter_mut() {
      for val in row.iter_mut() {
        *val = super::relu(*val);
      }
    }

    let (pool_out, _): (Matrix, Vec<Vec<(usize, usize)>>) =
      super::max_pool2x2(&conv_out);

    let flat: Vector = super::flatten(&pool_out);

    let fc_weights = &self.fc_weight_matrix[weight_index];

    super::perceptron(&flat, fc_weights, self.fc_bias)
  }
}
