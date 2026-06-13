use ::std::fs;
use ::std::io::Error;
use ::std::path::PathBuf;

const BYTES_PER_IMAGE: usize = 28 * 28;

const OFFSET: usize = 4 * 4;

#[derive(Default)]
pub struct Data {
  byte_vec: Vec<u8>,
}

impl Data {
  pub fn length(&self) -> usize {
    self.byte_vec.len()
  }

  pub fn load(&mut self) -> Result<(), Error> {
    let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

    let mut path: PathBuf = PathBuf::from(cargo_manifest_dir);

    path.push("archive");

    path.push("train-images-idx3-ubyte");

    path.push("train-images-idx3-ubyte");

    self.byte_vec = fs::read(path)?;

    Ok(())
  }

  pub fn record_count(&self) -> usize {
    let length = self.length();

    (length - OFFSET) / BYTES_PER_IMAGE
  }
}
