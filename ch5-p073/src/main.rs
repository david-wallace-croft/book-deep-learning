use ::rand::rngs::ThreadRng;
use ::rand_distr::{Distribution, Normal, Uniform};

fn main() {
  let mut rng = ThreadRng::default();

  let uniform = Uniform::new(-1., 1.).unwrap();

  let u_sample: f64 = uniform.sample(&mut rng);

  println!("Uniform sample: {u_sample}");

  let normal = Normal::new(0., 1.).unwrap();

  let n_sample: f64 = normal.sample(&mut rng);

  println!("Normal sample: {n_sample}");
}
