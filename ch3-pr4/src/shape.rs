use ::std::f64::consts;

pub enum Shape {
  Circle(f64),
  Rectangle(f64, f64),
}

impl Shape {
  pub fn area(s: Shape) -> f64 {
    match s {
      Shape::Circle(radius) => consts::PI * radius * radius,
      Shape::Rectangle(width, height) => width * height,
    }
  }
}
