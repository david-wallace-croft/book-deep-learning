use ::std::sync::mpsc::{self, Receiver, Sender};
use ::std::thread;

fn main() {
  let (log_tx, log_rx): (Sender<String>, Receiver<String>) = mpsc::channel();

  thread::spawn(move || {
    for message in log_rx {
      println!("[LOG] {message}");
    }
  });

  log_tx.send("Epoch 1: Loss = 0.25".to_string()).unwrap();
}
