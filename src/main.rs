use rand::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct Neuron {
    value: Box<RefCell<f32>>,
    weights: Box<Vec<(Rc<RefCell<Neuron>>, f32)>>,
    backprop: Box<Vec<(Rc<RefCell<Neuron>>)>>,
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Rc<RefCell<Neuron>>>,
    next_layer: RefCell<Option<Rc<Layer>>>,
}

#[derive(Debug)]
struct Model {
    start: RefCell<Rc<Layer>>,
    end: RefCell<Rc<Layer>>,
}

impl Neuron {
    fn new() -> Neuron {
        Neuron {
            value: Box::new(RefCell::new(0.0)),
            weights: Box::new(vec![]),
            backprop: Box::new(vec![]),
        }
    }

    fn fire(&self) {
        for (neuron, weight) in self.weights.iter() {
            neuron.borrow_mut().append(*self.value.borrow() * weight);
        }
    }

    fn append(&self, inp: f32) {
        *self.value.borrow_mut() += inp;
    }

    fn set(&self, inp: f32) {
        self.value.replace(inp);
    }
}

impl Layer {
    fn new(nodes: usize) -> Rc<Layer> {
        Rc::new(Layer {
            neurons: (0..nodes)
                .map(|_| Rc::new(RefCell::new(Neuron::new())))
                .collect(),
            next_layer: RefCell::new(None),
        })
    }

    fn neuron_get(&self) -> Vec<Rc<RefCell<Neuron>>> {
        self.neurons.clone()
    }

    fn connect(&self, b_layer: Rc<Layer>) {
        let mut random = rand::thread_rng();
        for b_neur in b_layer.neuron_get() {
            for t_neur in self.neuron_get() {
                t_neur
                    .borrow_mut()
                    .weights
                    .push((b_neur.clone(), random.gen()));
            }
        }
        self.next_layer.replace(Some(b_layer));
    }

    fn layer_size(&self) -> usize {
        self.neurons.len()
    }

    fn next_l(&self) -> Option<Rc<Layer>> {
        let r = self.next_layer.borrow();
        if let Some(real) = r.as_ref() {
            Some(real.clone())
        } else {
            None
        }
    }

    fn full_fire(&self) {
        for neuron in self.neurons.iter() {
            neuron.borrow().fire();
        }
        if let Some(layer) = self.next_l() {
            layer.full_fire()
        }
    }

    fn fill(&self, inp: &[f32]) {
        for (neuron, val) in self.neurons.iter().zip(inp) {
            neuron.borrow().set(*val);
        }
    }

    fn clear(&self) {
        for neuron in self.neurons.iter() {
            neuron.borrow().set(0.0);
        }
        if let Some(layer) = self.next_l() {
            layer.clear()
        }
    }
}

impl Model {
    fn new(start: Rc<Layer>, end: Rc<Layer>) -> Model {
        Model {
            start: RefCell::new(start),
            end: RefCell::new(end),
        }
    }

    fn evaluate(&self, inp: &[f32]) -> Vec<f32> {
        if inp.len() != self.start.borrow().layer_size() {
            panic!(
                "invalid size arr for first layer: expected {} but got {}",
                self.start.borrow().layer_size(),
                inp.len()
            );
        }

        self.start.borrow().clear();
        self.start.borrow().fill(inp);
        self.start.borrow().full_fire();
        self.end
            .borrow()
            .neuron_get()
            .iter()
            .map(|x| *(x.borrow().value.borrow()))
            .collect()
    }

    fn train(&self, inp: &[f32], out: &[f32]) {
        if inp.len() != self.start.borrow().layer_size() {
            panic!(
                "invalid size arr for first layer: expected {} but got {}",
                self.start.borrow().layer_size(),
                inp.len()
            );
        }

        if out.len() != self.end.borrow().layer_size() {
            panic!(
                "invalid size arr for last layer: expected {} but got {}",
                self.end.borrow().layer_size(),
                out.len()
            );
        }

        self.start.borrow().clear();
        self.start.borrow().fill(inp);
        self.start.borrow().full_fire();
        let pred_out: Vec<f32> = self
            .end
            .borrow()
            .neuron_get()
            .iter()
            .map(|x| *(x.borrow().value.borrow()))
            .collect();
        let full_mse: f32 = pred_out.iter().zip(out).fold(0.0, |accum, (pred, actual)| {
            accum + (actual - pred).powf(2.0)
        }) / self.end.borrow().layer_size() as f32;
        println!("mse: {}", full_mse);


    }
}

fn main() {
    let layer0 = Layer::new(15);
    let layer1 = Layer::new(5);
    layer0.connect(layer1.clone());
    let model = Model::new(layer0, layer1);
    let inp = (1..16).map(|x| x as f32).collect::<Vec<f32>>();
    let out = (0..5)
        .map(|x| (3 * x + 3 * x + 1 + 3 * x + 2) as f32)
        .collect::<Vec<f32>>();
    println!("{:?}", model.evaluate(&inp));
    model.train(&inp, &out);
}
