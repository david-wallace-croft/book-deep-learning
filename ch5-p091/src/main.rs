use ::plotters::prelude::*;
use ::rand::rngs::ThreadRng;
use ::rand_distr::{Distribution, Uniform};
use ::std::error::Error;
use ::std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
  let mut rng = ThreadRng::default();

  let uniform_0_to_5 = Uniform::new(0., 5.).unwrap();

  let uniform_5_to_10 = Uniform::new(5., 10.).unwrap();

  let cargo_manifest_dir: &str = env!("CARGO_MANIFEST_DIR");

  let mut path: PathBuf = PathBuf::from(cargo_manifest_dir);

  path.push("scatter_classified.png");

  let red_points: Vec<(f64, f64)> = (0..25)
    .map(|_| {
      (
        uniform_0_to_5.sample(&mut rng),
        uniform_0_to_5.sample(&mut rng),
      )
    })
    .collect();

  let blue_points: Vec<(f64, f64)> = (0..25)
    .map(|_| {
      (
        uniform_5_to_10.sample(&mut rng),
        uniform_5_to_10.sample(&mut rng),
      )
    })
    .collect();

  let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();

  root.fill(&WHITE)?;

  let mut chart = ChartBuilder::on(&root)
    .caption("Scatter Plot by Category", ("sans-serif", 30))
    .margin(20)
    .x_label_area_size(40)
    .y_label_area_size(40)
    .build_cartesian_2d(0.0..10., 0.0..10.)?;

  chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

  chart
    .draw_series(
      red_points
        .iter()
        .map(|(x, y)| Circle::new((*x, *y), 4, RED.filled())),
    )?
    .label("Red Class")
    .legend(|(x, y)| Circle::new((x, y), 4, RED.filled()));

  chart
    .draw_series(
      blue_points
        .iter()
        .map(|(x, y)| Circle::new((*x, *y), 4, BLUE.filled())),
    )?
    .label("Blue Class")
    .legend(|(x, y)| Circle::new((x, y), 4, BLUE.filled()));

  chart
    .configure_series_labels()
    .border_style(BLACK)
    .background_style(WHITE.mix(0.8))
    .draw()?;

  println!("Scatter plot saved to {path:?}");

  Ok(())
}
