use ::std::collections::{BTreeSet, HashMap};

#[derive(Default)]
pub struct Vocab {
  pub id_to_word: Vec<String>,
  pub word_to_id: HashMap<String, i64>,
}

impl Vocab {
  pub fn encode(
    &self,
    sentence: &str,
    sequence_length: usize,
  ) -> Vec<i64> {
    let mut ids: Vec<i64> = sentence
      .split_whitespace()
      .map(|word: &str| {
        let word_lowercase: String = word.to_lowercase();

        let id_option: Option<&i64> = self.word_to_id.get(&word_lowercase);

        let id_option_cloned: Option<i64> = id_option.cloned();

        id_option_cloned.unwrap_or(0)
      })
      .collect();

    ids.truncate(sequence_length);

    while ids.len() < sequence_length {
      ids.push(0);
    }

    ids
  }

  pub fn new(words: &[&str]) -> Self {
    let mut set: BTreeSet<String> = BTreeSet::new();

    set.insert("<unknown>".to_string());

    for word in words {
      let word_lowercase: String = word.to_lowercase();

      set.insert(word_lowercase);
    }

    let id_to_word: Vec<String> = set.into_iter().collect();

    let word_to_id: HashMap<String, i64> = id_to_word
      .iter()
      .enumerate()
      .map(|(index, word): (usize, &String)| (word.clone(), index as i64))
      .collect();

    Self {
      id_to_word,
      word_to_id,
    }
  }

  pub fn size(&self) -> i64 {
    self.id_to_word.len() as i64
  }
}
