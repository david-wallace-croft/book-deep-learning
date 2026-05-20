use ::std::sync::mpsc::{self, Receiver, SendError, Sender};
use ::std::thread;
use ::std::time::Duration;

fn main() {
  let (sender, receiver): (Sender<String>, Receiver<String>) = mpsc::channel();

  thread::spawn(move || {
    for i in 0..4 {
      thread::sleep(Duration::from_millis(100));

      let message: String = format!("Data {i}");

      let _result: Result<(), SendError<String>> = sender.send(message);
    }
  });

  for message in receiver {
    println!("Training on {message}");

    thread::sleep(Duration::from_millis(1_000));
  }

  println!("Done");
}
