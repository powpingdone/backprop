use ndarray::*;
use rand::prelude::*;
use std::boxed::Box;
use std::cell::RefCell;
use std::rc::Rc;

struct LayerStruct {
    vals: RefCell<Array1<f32>>,
    weights: RefCell<Array2<f32>>,
    next_layer: RefCell<Option<Layer>>,
    prev_layer: RefCell<Option<Layer>>,
}

type Layer = Box<Rc<LayerStruct>>;

struct Model {
    inp: RefCell<Layer>,
    out: RefCell<Layer>,
}

impl LayerStruct {
    fn new(neurons: usize) -> Layer {
        Box::new(Rc::new(LayerStruct {
            vals: RefCell::new((0..neurons).map(|_| 0.0).collect()),
            weights: RefCell::new(Array2::<f32>::zeros((0, 0))),
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
        for x in new_weights.iter_mut() {
            *x = rng.gen::<f32>() * 2.0;
        }
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
            self.next_l()
                .unwrap()
                .fill(&self.vals.borrow().dot(&self.weights.borrow().t()));
            layer.fire()
        }
    }

    fn fetch(&self) -> Array1<f32> {
        self.vals.borrow().clone()
    }

    fn fetch_weights(&self) -> Array2<f32> {
        self.weights.borrow().clone()
    }

    fn set_weights(&self, new_weights: Array2<f32>) {
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
        self.inp.borrow().clear();
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
        let (preds, mse) = self.predict(input);
        println!("preds: {:?}\n  mse: {:?}\n real: {:?}", preds, mse, output);
        let mut err: Array1<f32> = output - preds;
        let mut deltas: Vec<Array1<f32>> = vec![];
        for layer in layer_outs.iter().rev() {
            let delta = layer.fetch() * err.clone();
            println!("delta: {:?}\n  err: {:?}", delta, err);
            if let Some(_l) = layer.prev_l() {
                err = delta.dot(&layer.prev_l().unwrap().fetch_weights());
            } else {
                err = Array1::<f32>::zeros(1);
            }
            deltas.push(delta);
        }
        let deltas = deltas;
        for (layer, delta) in layer_outs.iter().rev().zip(deltas) {
            let new_weights = layer.fetch_weights() + (layer.fetch().t().dot(&delta) * LR);
            println!("new_weights: {:?}", new_weights);
            layer.set_weights(new_weights)
        }

        println!();
    }
}

fn main() {
    let x = LayerStruct::new(3);
    let y = LayerStruct::new(1);
    x.connect(y.clone());
    let model = Model::new(x, y);
    let mut rng = rand::thread_rng();
    let input: Vec<Array1<f32>> = (0..1000)
        .map(|_| Array1::from_vec((1..=3).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>()))
        .collect();
    let output: Vec<Array1<f32>> = (0..1000)
        .zip(&input)
        .map(|(_, inpvals)| Array1::from_vec(vec![inpvals.sum() / 3.0]))
        .collect();
    for _ in 0..10 {
        for i in 0..1000 {
            model.train(&input[i], &output[i]);
        }
    }

    println!(
        "input: {:?}\npreds: {:?}\n real: {:?}",
        input[0],
        model.predict(&input[0]),
        output[0]
    );
}
