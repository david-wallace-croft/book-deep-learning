#![expect(dead_code)]
#![expect(unused_imports)]
#![expect(unused_mut)]
#![expect(unused_variables)]

mod encoder_block;
mod example;
mod mhsa;
mod tiny_nlp_transformer;
mod vocab;

use self::example::Example;
use self::tiny_nlp_transformer::TinyNlpTransformer;
use self::vocab::Vocab;
use ::tch::nn::{Adam, Embedding, Optimizer, OptimizerConfig, Path, VarStore};
use ::tch::{Device, Kind, Result, Tensor};

const BATCH_SIZE: i64 = 4;
const D_FF: i64 = 128;
const D_MODEL: i64 = 64;
const EPOCHS: i64 = 300;
const LEARNING_RATE: f64 = 1e-3;
const N_HEADS: i64 = 4;
const SEQ_LEN: usize = 8;
const RANDOM_SEED: i64 = 42;

fn main() -> Result<()> {
  tch::manual_seed(RANDOM_SEED);

  let device: Device = Device::Cpu;

  let (vocab, dataset) = toy_data(SEQ_LEN);

  let n = dataset.len() as i64;

  let mut vs = VarStore::new(device);

  let root = &vs.root();

  let model =
    TinyNlpTransformer::new(root, vocab.size(), D_MODEL, N_HEADS, D_FF, device);

  todo!()
}

fn toy_data(seq_len: usize) -> (Vocab, Vec<Example>) {
  let pos = [
    "i love this movie",
    "this film is fantastic",
    "what a great experience",
    "absolutely wonderful and inspiring",
    "i really liked it",
  ];

  let neg = [
    "i hate this movie",
    "this film is terrible",
    "what a bad experience",
    "absolutely awful and boring",
    "i really disliked it",
  ];

  let words: Vec<&str> = pos
    .iter()
    .chain(neg.iter())
    .flat_map(|s| s.split_whitespace())
    .collect();

  let vocab = Vocab::new(&words);

  let mut data = Vec::new();

  for s in pos {
    data.push(Example {
      x: vocab.encode(s, SEQ_LEN),
      y: 1,
    });
  }

  for s in neg {
    data.push(Example {
      x: vocab.encode(s, SEQ_LEN),
      y: 0,
    });
  }

  (vocab, data)
}
