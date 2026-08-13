mod encoder_block;
mod example;
mod multi_head_self_attention;
mod tiny_nlp_transformer;
mod vocab;

use self::example::Example;
use self::tiny_nlp_transformer::TinyNlpTransformer;
use self::vocab::Vocab;
use ::tch::nn::{Adam, OptimizerConfig, VarStore};
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

  // Data

  let (vocab, dataset) = toy_data(SEQ_LEN);

  let n = dataset.len() as i64;

  // Model

  let vs = VarStore::new(device);

  let root = &vs.root();

  let model =
    TinyNlpTransformer::new(root, vocab.size(), D_MODEL, N_HEADS, D_FF, device);

  let mut opt = Adam::default().build(&vs, LEARNING_RATE).unwrap();

  // Training loop

  for epoch in 1..=EPOCHS {
    let mut loss_epoch = 0.;

    let mut acc_epoch = 0.;

    let mut i0 = 0;

    while i0 < n {
      let i1 = (i0 + BATCH_SIZE).min(n);

      let batch = &dataset[i0 as usize..i1 as usize];

      let x: Vec<i64> = batch.iter().flat_map(|ex| ex.x.clone()).collect();

      let y: Vec<i64> = batch.iter().map(|ex| ex.y).collect();

      let xs = Tensor::from_slice(&x).to(device).view([
        i1 - i0,
        SEQ_LEN as i64,
      ]);

      let ys = Tensor::from_slice(&y).to(device);

      let logits = model.forward(&xs, true);

      let loss = logits.cross_entropy_for_logits(&ys);

      let acc = accuracy(&logits, &ys);

      opt.backward_step(&loss);

      loss_epoch += loss.double_value(&[]);

      acc_epoch += acc * (i1 - i0) as f64;

      i0 = i1;
    }

    if epoch % 20 == 0 || epoch == 1 {
      println!(
        "epoch {:4} | loss {:.4} | acc {:.1}%",
        epoch,
        loss_epoch,
        100. * acc_epoch / (n as f64)
      );
    }
  }

  let test_sentences = [
    "i really love this fantastic movie",
    "this film is bad and boring",
    "great and inspiring experience",
    "absolutely terrible",
    "i disliked this",
    "i liked this wonderful film",
  ];

  println!("\n--- quick test ---");

  for s in test_sentences {
    let ids = vocab.encode(s, SEQ_LEN);

    let xs = Tensor::from_slice(&ids).to(device).view([
      1,
      SEQ_LEN as i64,
    ]);

    let logits = model.forward(&xs, false);

    let prob = logits.softmax(-1, Kind::Float);

    let cls = prob.argmax(-1, false).int64_value(&[]);

    let p_pos = prob.double_value(&[
      0, 1,
    ]);

    println!("{:45} -> class={} (p_pos={:.3})", s, cls, p_pos);
  }

  Ok(())
}

fn accuracy(
  logits: &Tensor,
  y: &Tensor,
) -> f64 {
  let pred = logits.argmax(-1, false);

  pred
    .eq_tensor(y)
    .to_kind(Kind::Float)
    .mean(Kind::Float)
    .double_value(&[])
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
      x: vocab.encode(s, seq_len),
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
