use ::csv::{Error, Reader, StringRecord};
use ::std::fs::File;
use ::std::path::PathBuf;

fn main() -> Result<(), Error> {
  let records = read_records()?;

  for record in records {
    println!("{record:?}");
  }

  Ok(())
}

fn read_records() -> Result<Vec<StringRecord>, Error> {
  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let mut path = PathBuf::from(cargo_manifest_dir);

  path.push("data.csv");

  let file = File::open(path)?;

  let mut reader = Reader::from_reader(file);

  reader.records().into_iter().collect()
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test() -> Result<(), Error> {
    let expected: &str = concat!(
      r#"[StringRecord(["1", " 2", " 3"]),"#,
      r#" StringRecord(["4", " 5", " 6"])]"#
    );

    let records: Vec<StringRecord> = read_records()?;

    let actual: String = format!("{records:?}");

    assert_eq!(actual, expected);

    Ok(())
  }
}
