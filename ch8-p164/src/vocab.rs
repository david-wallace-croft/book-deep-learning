use ::std::collections::HashMap;

#[derive(Default)]
pub struct Vocab {
  itos: Vec<String>,
  stoi: HashMap<String, i64>,
}
