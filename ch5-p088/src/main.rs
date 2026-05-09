use ::plotters::element::PathElement;
use ::plotters::prelude::*;
use ::rand::rngs::ThreadRng;
use ::rand_distr::{Distribution, Uniform};
use ::std::error::Error;
use ::std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
  let mut rng = ThreadRng::default();

  let uniform = Uniform::new(-0.01, 0.01).unwrap();

  let loss_values: Vec<f64> = (0..100)
    .map(|epoch| {
      let base_loss = 1. / (epoch as f64 + 1.);

      base_loss + uniform.sample(&mut rng)
    })
    .collect();

  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let mut path: PathBuf = PathBuf::from(cargo_manifest_dir);

  path.push("training_loss.png");

  let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();

  root.fill(&WHITE)?;

  let max_loss = loss_values.iter().cloned().fold(f64::NAN, f64::max);

  let mut chart = ChartBuilder::on(&root)
    .caption("Simulated Training Loss", ("sans-serif", 10))
    .margin(20)
    .x_label_area_size(40)
    .y_label_area_size(50)
    .build_cartesian_2d(0..100, 0.0..max_loss)?;

  chart
    .configure_mesh()
    .x_desc("Epoch")
    .y_desc("Loss")
    .draw()?;

  chart
    .draw_series(LineSeries::new(
      loss_values.iter().enumerate().map(|(x, y)| (x as i32, *y)),
      &BLUE,
    ))?
    .label("Loss")
    .legend(|(x, y)| {
      PathElement::new(
        vec![
          (x, y),
          (x + 20, y),
        ],
        BLUE,
      )
    });

  chart
    .configure_series_labels()
    .background_style(WHITE.mix(0.8))
    .border_style(BLACK)
    .draw()?;

  println!("Loss plot saved to training_loss.png");

  Ok(())
}
