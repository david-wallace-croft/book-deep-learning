use ::std::sync::{Arc, Barrier};
use ::std::thread;
use ::std::time::Duration;

fn main() {
  let barrier = Arc::new(Barrier::new(3));

  //

  let b1 = barrier.clone();

  let t1 = thread::spawn(move || {
    compute_layer("Layer 1", b1);
  });

  //

  let b2 = barrier.clone();

  let t2 = thread::spawn(move || {
    compute_layer("Layer 2", b2);
  });

  //

  let b3 = barrier.clone();

  let t3 = thread::spawn(move || {
    compute_layer("Layer 3", b3);
  });

  //

  t1.join().unwrap();

  t2.join().unwrap();

  t3.join().unwrap();

  println!("All layers completed.");
}

fn compute_layer(
  name: &str,
  barrier: Arc<Barrier>,
) {
  println!("Computing layer: {name}");

  thread::sleep(Duration::from_millis(100));

  println!("Layer {name} done.");

  barrier.wait();
}
