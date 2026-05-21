use ::std::sync::mpsc::{self, Receiver, SendError, Sender};
use ::std::thread;
use ::std::time::{Duration, Instant};

const DATA_COUNT: usize = 4;

const DATA_LOAD_DELAY_MILLIS: u64 = 100;

const DATA_TRAIN_DELAY_MILLIS: u64 = 1_000;

fn main() {
  let start: Instant = Instant::now();

  run_parallel();

  let duration_parallel: Duration = start.elapsed();

  let start: Instant = Instant::now();

  run_serial();

  let duration_serial: Duration = start.elapsed();

  println!("Parallel time ...: {duration_parallel:?}");

  println!("Serial time .....: {duration_serial:?}");
}

fn run_parallel() {
  let (sender, receiver): (Sender<String>, Receiver<String>) = mpsc::channel();

  thread::spawn(move || {
    for i in 1..=DATA_COUNT {
      thread::sleep(Duration::from_millis(DATA_LOAD_DELAY_MILLIS));

      let message: String = format!("Data {i}");

      let _result: Result<(), SendError<String>> = sender.send(message);
    }
  });

  for message in receiver {
    println!("Training on {message}");

    thread::sleep(Duration::from_millis(DATA_TRAIN_DELAY_MILLIS));
  }
}

fn run_serial() {
  for i in 1..=DATA_COUNT {
    thread::sleep(Duration::from_millis(DATA_LOAD_DELAY_MILLIS));

    let message: String = format!("Data {i}");

    println!("Training on {message}");

    thread::sleep(Duration::from_millis(DATA_TRAIN_DELAY_MILLIS));
  }
}
