use self::shape::Shape;

mod shape;

const HEIGHT: f64 = 2.5;
const RADIUS: f64 = 3.;
const WIDTH: f64 = 4.;

const CIRCLE: Shape = Shape::Circle(RADIUS);
const RECTANGLE: Shape = Shape::Rectangle(WIDTH, HEIGHT);

fn main() {
  println!(
    "The area of a Circle with radius {} is {}.",
    RADIUS,
    Shape::area(CIRCLE),
  );

  println!(
    "The area of a Rectangle with width {} and height {} is {}.",
    WIDTH,
    HEIGHT,
    Shape::area(RECTANGLE),
  );
}

#[cfg(test)]
mod test {
  use super::*;

  const DELTA: f64 = 0.01;
  const EXPECTED_CIRCLE_AREA: f64 = 28.27;
  const EXPECTED_RECTANGLE_AREA: f64 = 10.;

  #[test]
  fn test() {
    assert!((Shape::area(CIRCLE) - EXPECTED_CIRCLE_AREA).abs() < DELTA);

    assert!((Shape::area(RECTANGLE) - EXPECTED_RECTANGLE_AREA).abs() < DELTA);
  }
}
