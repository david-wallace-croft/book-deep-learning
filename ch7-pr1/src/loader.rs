use super::data::Data;
use ::std::fs;
use ::std::io::Error;
use ::std::path::PathBuf;

const BYTES_PER_COLUMN: usize = 28;

const BYTES_PER_ROW: usize = 28;

const OFFSET: usize = 4 * 4;

#[derive(Default)]
pub struct Loader {
  // TODO
}

impl Loader {
  pub fn load(&self) -> Result<Data, Error> {
    let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

    let mut path: PathBuf = PathBuf::from(cargo_manifest_dir);

    path.push("archive");

    path.push("train-images-idx3-ubyte");

    path.push("train-images-idx3-ubyte");

    let byte_vec: Vec<u8> = fs::read(path)?;

    let mut images: Vec<Vec<Vec<f32>>> = Default::default();

    let mut index: usize = OFFSET;

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

      images.push(image_vec);
    }

    Ok(Data {
      images,
    })
  }
}
