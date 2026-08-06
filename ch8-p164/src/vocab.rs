use ::std::collections::{BTreeSet, HashMap};

#[derive(Default)]
pub struct Vocab {
  pub itos: Vec<String>,
  pub stoi: HashMap<String, i64>,
}

impl Vocab {
  pub fn encode(
    &self,
    sentence: &str,
    seq_len: usize,
  ) -> Vec<i64> {
    let mut ids: Vec<i64> = sentence
      .split_whitespace()
      .map(|w| self.stoi.get(&w.to_lowercase()).cloned().unwrap_or(0))
      .collect();

    ids.truncate(seq_len);

    while ids.len() < seq_len {
      ids.push(0);
    }

    ids
  }

  pub fn new(words: &[&str]) -> Self {
    let mut set = BTreeSet::new();

    set.insert("<unk>".to_string());

    for w in words {
      set.insert(w.to_lowercase());
    }

    let itos: Vec<String> = set.into_iter().collect();

    let stoi: HashMap<String, i64> = itos
      .iter()
      .enumerate()
      .map(|(i, s)| (s.clone(), i as i64))
      .collect();

    Self {
      itos,
      stoi,
    }
  }

  pub fn size(&self) -> i64 {
    self.itos.len() as i64
  }
}
