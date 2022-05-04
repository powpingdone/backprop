use rand::prelude::*;
use std::boxed::Box;
use std::cell::RefCell;
use std::rc::Rc;
use ndarray::*;

#[derive(Debug)]
struct LayerStruct {
    vals: RefCell<Array1<f32>>,
    weights: RefCell<Array2<f32>>,
    next_layer: RefCell<Option<Layer>>,
    prev_layer: RefCell<Option<Layer>>,
}

type Layer = Box<Rc<LayerStruct>>;

#[derive(Debug)]
struct Model {
    inp: RefCell<Layer>,
    out: RefCell<Layer>,
}

impl LayerStruct {
    fn new(neurons: usize) -> Layer {
        Box::new(Rc::new(LayerStruct {
            vals: RefCell::new((0..neurons).map(|_| 0.0).collect()),
            weights: RefCell::new(Array2::<f32>::zeros((0,0))),
            next_layer: RefCell::new(None),
            prev_layer: RefCell::new(None),
        }))
    }

    fn neuron_amt(&self) -> usize {
        self.vals.borrow().len()
    }

    fn connect(&self, layer: Layer) {
        self.next_layer.replace(Some(layer.clone()));
        let mut rng = rand::thread_rng();
        let mut new_weights = Array2::<f32>::zeros((layer.neuron_amt(), self.neuron_amt()));
        new_weights.iter_mut().map(|x|*x = 2.0*rng.gen::<f32>()).collect::<()>();
        self.weights.replace(new_weights);
    }

    fn connect_prev(&self, layer: Layer) {
        self.prev_layer.replace(Some(layer));
    }

    fn next_l(&self) -> Option<Layer> {
        if let Some(layer) = self.next_layer.borrow().clone() {
            Some(layer)
        } else {
            None
        }
    }

    fn prev_l(&self) -> Option<Layer> {
        if let Some(layer) = self.prev_layer.borrow().clone() {
            Some(layer)
        } else {
            None
        }
    }

    fn clear(&self) {
        let mut neurons = self.vals.borrow_mut();
        for neuron in neurons.iter_mut() {
            *neuron = 0.0;
        }

        if let Some(layer) = self.next_l() {
            layer.clear();
        }
    }

    fn fill(&self, input: &Array1<f32>) {
        if input.len() != self.neuron_amt() {
            panic!(
                "input has len {} but neuron length is {}",
                input.len(),
                self.neuron_amt()
            );
        }

        for (val, next) in self.vals.borrow_mut().iter_mut().zip(input) {
            *val = *next;
        }
    }

    fn fire(&self) {
        if let Some(layer) = self.next_l() {
            self.next_l().unwrap().fill(&self.vals.borrow().dot(&*self.weights.borrow()));
            layer.fire()
        }
    }

    fn fetch(&self) -> Array1<f32> {
        self.vals.borrow().clone()
    }

    fn fetch_weights(&self) -> Array2<f32> {
        self.weights.borrow().clone()
    }

    fn set_weights(&self, new_weights:Array2<f32>) {
        self.weights.replace(new_weights);
    }
}

impl Model {
    fn new(inp: Layer, out: Layer) -> Model {
        let mut layers = vec![inp.clone()];
        let mut curr = inp.clone();
        while let Some(layer) = curr.next_l().clone() {
            curr = layer.clone();
            layers.push(layer);
        }

        let mut curr = out.clone();
        while let Some(layer) = layers.pop() {
            curr.connect_prev(layer.clone());
            curr = layer;
        }

        Model {
            inp: RefCell::new(inp),
            out: RefCell::new(out),
        }
    }

    fn predict(&self, input: &Array1<f32>) -> (Array1<f32>, f32) {
        self.inp.borrow().fill(input);
        self.inp.borrow().fire();
        let preds = self.out.borrow().fetch();
        let mse = preds
            .iter()
            .zip(input)
            .fold(0.0, |accum, (pred, actual)| accum + (actual - pred).powi(2))
            / preds.len() as f32;
        (preds, mse)
    }

    fn train(&self, input: &Array1<f32>, output: &Array1<f32>) {
        const LR: f32 = 0.01;
        let mut layer_outs = vec![self.inp.borrow().clone()];
        let mut curr = self.inp.borrow().clone();
        while let Some(layer) = curr.next_l().clone() {
            curr = layer.clone();
            layer_outs.push(layer);
        }
        let layer_outs = layer_outs;
        let (preds, _) = self.predict(input);
        let mut err: Array1<f32> = output - preds;
        let mut deltas: Vec<Array1<f32>> = vec![];
        for layer in layer_outs.iter().rev().skip(1) {
            let delta = layer.fetch() * err;
            if let Some(_l) = layer.prev_l() {
                err = delta.dot(&layer.fetch_weights().t());
            } else {
                err = Array1::<f32>::zeros(1);
            }
            deltas.push(delta);
        }
        let deltas = deltas;
        for (layer, delta) in layer_outs.iter().rev().skip(1).zip(deltas) {
            layer.set_weights(layer.fetch_weights() + (layer.fetch().t().dot(&delta) * LR));
        }
    }
}

fn main() {
    let x = LayerStruct::new(15);
    let y = LayerStruct::new(5);
    x.connect(y.clone());
    let model = Model::new(x, y);
    let input = (1..=15).map(|x| x as f32).collect();
    let output = (1..=5).map(|x| (x + x * 2 + x * 3) as f32 ).collect();
    for _ in 0..100000 {
        model.train(&input, &output);
    }
    println!("{:?}", model);
    println!("{:?}", model.predict(&input));
}
