use ndarray::*;
use rand::prelude::*;
use std::boxed::Box;
use std::cell::RefCell;
use std::rc::Rc;

trait ZORBLayer<T, DIN: Dimension, DOUT: Dimension> {
    fn forward(&self, x: Array<T, DIN>) -> Array<T, DOUT>;
    fn backward(&self, y: Array<T, DOUT>) -> Array<T, DIN>;
    fn update(&self, x: Array<T, DIN>, y: Array<T, DOUT>);
}

struct Dense<T> {
    weights: RefCell<Array2<T>>
}

fn main() {}
