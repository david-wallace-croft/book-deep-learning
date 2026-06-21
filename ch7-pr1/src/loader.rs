use super::aliases::{Category, Dataset, Image};
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
  pub length: usize,
  pub images_path_buf: PathBuf,
  pub labels_path_buf: PathBuf,
}

impl Loader {
  pub fn default_test_data_loader() -> Loader {
    let images_path_buf: PathBuf =
      Loader::make_path_buf("t10k-images.idx3-ubyte");

    let labels_path_buf: PathBuf =
      Loader::make_path_buf("t10k-labels.idx1-ubyte");

    Loader {
      length: DATASET_LENGTH_TEST,
      images_path_buf,
      labels_path_buf,
    }
  }

  pub fn default_train_data_loader() -> Loader {
    let images_path_buf: PathBuf =
      Loader::make_path_buf("train-images.idx3-ubyte");

    let labels_path_buf: PathBuf =
      Loader::make_path_buf("train-labels.idx1-ubyte");

    Loader {
      length: DATASET_LENGTH_TRAIN,
      images_path_buf,
      labels_path_buf,
    }
  }

  pub fn load(&self) -> Result<Dataset, Error> {
    let images: Vec<Image> = self.load_images()?;

    let labels: Vec<Category> = self.load_labels()?;

    let dataset: Dataset = images.into_iter().zip(labels).collect();

    Ok(dataset)
  }

  fn load_images(&self) -> Result<Vec<Image>, Error> {
    let byte_vec: Vec<u8> = fs::read(&self.images_path_buf)?;

    let mut images: Vec<Image> = Default::default();

    let mut index: usize = OFFSET_IMAGES;

    for _image_index in 0..self.length {
      let mut image: Image = Default::default();

      for _row_index in 0..BYTES_PER_ROW {
        let mut row_vec: Vec<f32> = Default::default();

        for _column_index in 0..BYTES_PER_COLUMN {
          let byte_at_index: &u8 = byte_vec.get(index).unwrap();

          let scaled_value: f32 = *byte_at_index as f32 / 255.;

          row_vec.push(scaled_value);

          index += 1;
        }

        image.push(row_vec);
      }

      images.push(image);
    }

    Ok(images)
  }

  fn load_labels(&self) -> Result<Vec<Category>, Error> {
    let mut labels: Vec<Category> = fs::read(&self.labels_path_buf)?;

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

  pub fn print_image(image: &Image) {
    for row_vec in image {
      for value in row_vec {
        let symbol = if *value >= 0.5 {
          '*'
        } else {
          '.'
        };

        print!("{symbol} ");
      }

      println!();
    }

    println!("===");
  }
}
