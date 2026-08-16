use self::example::Example;
use self::tiny_nlp_transformer::TinyNlpTransformer;
use self::vocab::Vocab;
use ::tch::nn::{Adam, Optimizer, OptimizerConfig, Path, VarStore};
use ::tch::{Device, Kind, Result, Tensor};

mod encoder_block;
mod example;
mod multi_head_self_attention;
mod tiny_nlp_transformer;
mod vocab;

const BATCH_SIZE: i64 = 4;
const FEED_FORWARD_DIMENSIONS: i64 = 128;
const EPOCHS: usize = 300;
const HEADS: i64 = 4;
const LEARNING_RATE: f64 = 1e-3;
const MODEL_DIMENSIONS: i64 = 64;
const PRINT_INTERVAL: usize = 10;
const RANDOM_SEED: i64 = 42;
const SENTENCES_NEGATIVE: [&str; 5] = [
  "i hate this movie",
  "this film is terrible",
  "what a bad experience",
  "absolutely awful and boring",
  "i really disliked it",
];
const SENTENCES_POSITIVE: [&str; 5] = [
  "i love this movie",
  "this film is fantastic",
  "what a great experience",
  "absolutely wonderful and inspiring",
  "i really liked it",
];
const SENTENCES_TEST: [&str; 6] = [
  "i really love this fantastic movie",
  "this film is bad and boring",
  "great and inspiring experience",
  "absolutely terrible",
  "i disliked this",
  "i liked this wonderful film",
];
const SEQUENCE_LENGTH: usize = 8;

fn main() -> Result<()> {
  tch::manual_seed(RANDOM_SEED);

  let device: Device = Device::Cpu;

  // Data

  let (vocab, dataset): (Vocab, Vec<Example>) = toy_data(SEQUENCE_LENGTH);

  let dataset_length: i64 = dataset.len() as i64;

  let vocabulary_size = vocab.size();

  // Model

  let var_store: VarStore = VarStore::new(device);

  let root_path: &Path<'_> = &var_store.root();

  let model: TinyNlpTransformer = TinyNlpTransformer::new(
    root_path,
    vocabulary_size,
    MODEL_DIMENSIONS,
    HEADS,
    FEED_FORWARD_DIMENSIONS,
    device,
  );

  // Adaptive Moment Estimation (Adam)
  // https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html
  let mut optimizer: Optimizer =
    Adam::default().build(&var_store, LEARNING_RATE).unwrap();

  // Training loop

  for epoch in 1..=EPOCHS {
    let mut loss: f64 = 0.;

    let mut accuracy_epoch: f64 = 0.;

    let mut i0: i64 = 0;

    while i0 < dataset_length {
      let i1: i64 = (i0 + BATCH_SIZE).min(dataset_length);

      let batch: &[Example] = &dataset[i0 as usize..i1 as usize];

      let x: Vec<i64> = batch.iter().flat_map(|ex| ex.x.clone()).collect();

      let y: Vec<i64> = batch.iter().map(|ex| ex.y).collect();

      let xs: Tensor = Tensor::from_slice(&x).to(device).view([
        i1 - i0,
        SEQUENCE_LENGTH as i64,
      ]);

      let ys: Tensor = Tensor::from_slice(&y).to(device);

      let logits: Tensor = model.forward(&xs, true);

      let loss_tensor: Tensor = logits.cross_entropy_for_logits(&ys);

      let accuracy: f64 = accuracy_from_logits(&logits, &ys);

      optimizer.backward_step(&loss_tensor);

      loss += loss_tensor.double_value(&[]);

      accuracy_epoch += accuracy * (i1 - i0) as f64;

      i0 = i1;
    }

    if epoch % PRINT_INTERVAL == 0 {
      let accuracy: f64 = 100. * accuracy_epoch / (dataset_length as f64);

      println!("epoch {epoch} | loss {loss:.3} | accuracy {accuracy:.1}%");
    }
  }

  println!("\n--- quick test ---");

  for test_sentence in SENTENCES_TEST {
    let ids: Vec<i64> = vocab.encode(test_sentence, SEQUENCE_LENGTH);

    let xs: Tensor = Tensor::from_slice(&ids).to(device).view([
      1,
      SEQUENCE_LENGTH as i64,
    ]);

    let logits: Tensor = model.forward(&xs, false);

    let prob: Tensor = logits.softmax(-1, Kind::Float);

    let cls: i64 = prob.argmax(-1, false).int64_value(&[]);

    let p_pos: f64 = prob.double_value(&[
      0, 1,
    ]);

    println!("{:45} -> class={} (p_pos={:.3})", test_sentence, cls, p_pos);
  }

  Ok(())
}

fn accuracy_from_logits(
  logits: &Tensor,
  y: &Tensor,
) -> f64 {
  // Returns the indices of the max value of all elements in the input tensor
  // https://docs.pytorch.org/docs/main/generated/torch.argmax.html
  // The first argument is the dimension to reduce
  let pred: Tensor = logits.argmax(-1, false);

  let correct_bool: Tensor = pred.eq_tensor(y);

  // println!("correct_bool: {correct_bool}");

  let correct_float: Tensor = correct_bool.to_kind(Kind::Float);

  // println!("correct_float: {correct_float}");

  let correct: Tensor = correct_float.mean(Kind::Float);

  // correct is a scalar
  // println!("correct: {correct}");

  // Returns a double value on tensors holding a single element
  correct.double_value(&[])
}

fn toy_data(sequence_length: usize) -> (Vocab, Vec<Example>) {
  let words: Vec<&str> = SENTENCES_POSITIVE
    .iter()
    .chain(SENTENCES_NEGATIVE.iter())
    .flat_map(|s: &&str| s.split_whitespace())
    .collect();

  let vocab: Vocab = Vocab::new(&words);

  let mut data: Vec<Example> = Vec::new();

  for s in SENTENCES_POSITIVE {
    data.push(Example {
      x: vocab.encode(s, sequence_length),
      y: 1,
    });
  }

  for s in SENTENCES_NEGATIVE {
    data.push(Example {
      x: vocab.encode(s, SEQUENCE_LENGTH),
      y: 0,
    });
  }

  (vocab, data)
}
