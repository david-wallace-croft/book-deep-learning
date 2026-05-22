use ::ndarray::{Array2, array, s};

fn main() {
  let input = array![
    [
      1., 2., 3., 4.
    ],
    [
      5., 6., 7., 8.
    ],
    [
      9., 10., 11., 12.
    ],
    [
      13., 14., 15., 16.
    ],
  ];

  let kernel = array![
    [
      1., 0.
    ],
    [
      0., -1.
    ],
  ];

  let output = convolve2d(&input, &kernel);

  println!("{output}");
}

fn convolve2d(
  input: &Array2<f64>,
  kernel: &Array2<f64>,
) -> Array2<f64> {
  let (h, w) = input.dim();

  let (k_h, k_w) = kernel.dim();

  let out_h = h - k_h + 1;

  let out_w = w - k_w + 1;

  let mut output = Array2::<f64>::zeros((out_h, out_w));

  for i in 0..out_h {
    for j in 0..out_w {
      let window = input.slice(s![
        i..i + k_h,
        j..j + k_w
      ]);

      let sum = (&window * kernel).sum();

      output[(i, j)] = sum;
    }
  }

  output
}
