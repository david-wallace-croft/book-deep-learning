use ::std::fs;
use ::std::io::Error;
use ::std::path::PathBuf;

pub type Image = Vec<Vec<f32>>;

pub type Category = f32;

pub type Dataset = Vec<(Image, Category)>;

// https://web.archive.org/web/20020622183530/http://yann.lecun.com/exdb/mnist/

const BYTES_PER_COLUMN: usize = 28;

const BYTES_PER_ROW: usize = 28;

const OFFSET_IMAGES: usize = 4 * 4;

const OFFSET_LABELS: usize = 2 * 4;

#[derive(Default)]
pub struct Loader {
  // TODO
}

impl Loader {
  pub fn load(&self) -> Result<Dataset, Error> {
    let train_images: Vec<Vec<Vec<f32>>> = self.load_train_images()?;

    let train_labels: Vec<u8> = self.load_train_labels()?;

    let dataset: Dataset = train_images
      .into_iter()
      .zip(train_labels)
      .map(|(image, category)| (image, category as f32))
      .collect();

    Ok(dataset)
  }

  fn load_train_images(&self) -> Result<Vec<Vec<Vec<f32>>>, Error> {
    let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

    let mut path: PathBuf = PathBuf::from(cargo_manifest_dir);

    path.push("archive");

    path.push("train-images.idx3-ubyte");

    let byte_vec: Vec<u8> = fs::read(path)?;

    let mut train_images: Vec<Vec<Vec<f32>>> = Default::default();

    let mut index: usize = OFFSET_IMAGES;

    for _image_index in 0..60_000 {
      let mut image_vec: Vec<Vec<f32>> = Default::default();

      for _row_index in 0..BYTES_PER_ROW {
        let mut row_vec: Vec<f32> = Default::default();

        for _column_index in 0..BYTES_PER_COLUMN {
          let byte_at_index = byte_vec.get(index).unwrap();

          let scaled_value = *byte_at_index as f32 / 255.;

          row_vec.push(scaled_value);

          index += 1;
        }

        image_vec.push(row_vec);
      }

      train_images.push(image_vec);
    }

    Ok(train_images)
  }

  fn load_train_labels(&self) -> Result<Vec<u8>, Error> {
    let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

    let mut path: PathBuf = PathBuf::from(cargo_manifest_dir);

    path.push("archive");

    path.push("train-labels.idx1-ubyte");

    let mut train_labels: Vec<u8> = fs::read(path)?;

    train_labels.drain(0..OFFSET_LABELS);

    Ok(train_labels)
  }
}
