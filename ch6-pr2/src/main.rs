use ::rand::rngs::ThreadRng;
use ::rand_distr::{Distribution, Normal};
use ::std::sync::{Arc, Barrier};
use ::std::thread::{self, JoinHandle};
use ::std::time::Duration;
use std::u64;

const DELAY_MEAN: f64 = 100.;
const DELAY_SDEV: f64 = 20.;
const LAYER_COUNT: usize = 4;
const PHASE_COUNT: usize = 3;

fn main() {
  let barrier: Arc<Barrier> = Arc::new(Barrier::new(LAYER_COUNT));

  (1..=LAYER_COUNT)
    .map(|layer: usize| (layer, barrier.clone()))
    .map(|(layer, barrier): (usize, Arc<Barrier>)| {
      thread::spawn(move || compute_layer(barrier, layer))
    })
    .collect::<Vec<JoinHandle<()>>>()
    .into_iter()
    .for_each(|join_handle: JoinHandle<()>| join_handle.join().unwrap());

  println!("All phases completed.");
}

fn compute_layer(
  barrier: Arc<Barrier>,
  layer: usize,
) {
  let name = format!("Layer {layer}");

  let normal: Normal<f64> = Normal::new(DELAY_MEAN, DELAY_SDEV).unwrap();

  let mut rng: ThreadRng = ThreadRng::default();

  for phase in 1..=PHASE_COUNT {
    let processing_delay: u64 = normal.sample(&mut rng).max(0.) as u64;

    println!("{name} starting phase {phase}");

    thread::sleep(Duration::from_millis(processing_delay));

    println!("{name} finished phase {phase} after {processing_delay} ms");

    barrier.wait();
  }
}
