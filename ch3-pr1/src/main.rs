fn main() {
  let text: String = "Hello, World!".to_string();

  consume_text(&text);

  println!("Input string: {text}");
}

fn consume_text(text: &str) {
  println!("Input string: {text}");
}
