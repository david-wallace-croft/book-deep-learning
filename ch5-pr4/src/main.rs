use ::plotters::prelude::*;
use ::std::error::Error;
use ::std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let mut path: PathBuf = PathBuf::from(cargo_manifest_dir);

  path.push("relu.png");

  let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();

  root.fill(&WHITE)?;

  let mut chart = ChartBuilder::on(&root)
    .caption("Rectified Linear Unit (ReLU)", ("sans-serif", 10))
    .margin(20)
    .x_label_area_size(40)
    .y_label_area_size(50)
    .build_cartesian_2d(-6.0..6.0, -6.0..6.0)?;

  chart
    .configure_mesh()
    .x_desc("Input")
    .y_desc("Output")
    .draw()?;

  chart
    .draw_series(LineSeries::new(
      (-50..=50).map(|x| x as f64 / 10.0).map(|x| (x, relu(x))),
      BLUE,
    ))?
    .label("Output");
  // .legend(|(x, y)| {
  //   PathElement::new(
  //     vec![
  //       (x, y),
  //       (x + 20, y),
  //     ],
  //     BLUE,
  //   )
  // });

  // chart
  //   .configure_series_labels()
  //   .background_style(WHITE.mix(0.8))
  //   .border_style(BLACK)
  //   .draw()?;

  Ok(())
}

fn relu(x: f64) -> f64 {
  if x.is_sign_positive() {
    x
  } else {
    0.
  }
}
