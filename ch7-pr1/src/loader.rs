use super::data::{Category, Data, Dataset, Image};
use ::std::fs;
use ::std::io::Error;
use ::std::path::PathBuf;

// https://web.archive.org/web/20020622183530/http://yann.lecun.com/exdb/mnist/

const BYTES_PER_COLUMN: usize = 28;

const BYTES_PER_ROW: usize = 28;

const OFFSET_IMAGES: usize = 4 * 4;

const OFFSET_LABELS: usize = 2 * 4;

const DATASET_LENGTH_TEST: usize = 10_000;

const DATASET_LENGTH_TRAIN: usize = 60_000;

pub struct Loader {
  test_data_length: usize,
  test_images_path_buf: PathBuf,
  test_labels_path_buf: PathBuf,
  train_data_length: usize,
  train_images_path_buf: PathBuf,
  train_labels_path_buf: PathBuf,
}

impl Loader {
  pub fn load(&self) -> Result<Data, Error> {
    let test_dataset: Dataset = self.load_dataset(true)?;

    let train_dataset: Dataset = self.load_dataset(false)?;

    let data: Data = Data {
      test_dataset,
      train_dataset,
    };

    Ok(data)
  }

  fn load_dataset(
    &self,
    use_test_data: bool,
  ) -> Result<Dataset, Error> {
    let images: Vec<Image> = self.load_images(use_test_data)?;

    let labels: Vec<Category> = self.load_labels(use_test_data)?;

    let dataset: Dataset = images.into_iter().zip(labels).collect();

    Ok(dataset)
  }

  fn load_images(
    &self,
    use_test_data: bool,
  ) -> Result<Vec<Image>, Error> {
    let path_buf: &PathBuf = if use_test_data {
      &self.test_images_path_buf
    } else {
      &self.train_images_path_buf
    };

    let data_length: usize = if use_test_data {
      self.test_data_length
    } else {
      self.train_data_length
    };

    let byte_vec: Vec<u8> = fs::read(path_buf)?;

    let mut images: Vec<Image> = Default::default();

    let mut index: usize = OFFSET_IMAGES;

    for _image_index in 0..data_length {
      let mut image_vec: Image = Default::default();

      for _row_index in 0..BYTES_PER_ROW {
        let mut row_vec: Vec<f32> = Default::default();

        for _column_index in 0..BYTES_PER_COLUMN {
          let byte_at_index: &u8 = byte_vec.get(index).unwrap();

          let scaled_value: f32 = *byte_at_index as f32 / 255.;

          row_vec.push(scaled_value);

          index += 1;
        }

        image_vec.push(row_vec);
      }

      images.push(image_vec);
    }

    Ok(images)
  }

  fn load_labels(
    &self,
    use_test_data: bool,
  ) -> Result<Vec<Category>, Error> {
    let path_buf: &PathBuf = if use_test_data {
      &self.test_labels_path_buf
    } else {
      &self.train_labels_path_buf
    };

    let mut labels: Vec<Category> = fs::read(path_buf)?
      .into_iter()
      .map(|c: u8| c as Category)
      .collect();

    labels.drain(0..OFFSET_LABELS);

    Ok(labels)
  }

  fn make_path_buf(filename: &'static str) -> PathBuf {
    let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

    let mut path_buf: PathBuf = PathBuf::from(cargo_manifest_dir);

    path_buf.push("archive");

    path_buf.push(filename);

    path_buf
  }
}

impl Default for Loader {
  fn default() -> Self {
    let test_images_path_buf: PathBuf =
      Loader::make_path_buf("t10k-images.idx3-ubyte");

    let test_labels_path_buf: PathBuf =
      Loader::make_path_buf("t10k-labels.idx1-ubyte");

    let train_images_path_buf: PathBuf =
      Loader::make_path_buf("train-images.idx3-ubyte");

    let train_labels_path_buf: PathBuf =
      Loader::make_path_buf("train-labels.idx1-ubyte");

    Self {
      test_data_length: DATASET_LENGTH_TEST,
      test_images_path_buf,
      test_labels_path_buf,
      train_data_length: DATASET_LENGTH_TRAIN,
      train_images_path_buf,
      train_labels_path_buf,
    }
  }
}
