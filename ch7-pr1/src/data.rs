pub type Image = Vec<Vec<f32>>;

pub type Category = f32;

pub type Dataset = Vec<(Image, Category)>;

pub struct Data {
  pub test_dataset: Dataset,
  pub train_dataset: Dataset,
}
