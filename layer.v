module nn

import mlx

// layer.v — the Layer sum type unifying every network building block, plus
// the dispatch methods used by Sequential and the optimizers.
//
// NOTE: this V version mis-parses interface method implementations in any
// module that imports mlx (a module with #flag/#include directives), so the
// framework uses a sum type with match dispatch instead of an interface.
// New layer types are registered by adding them to the sum type below and to
// each match arm.
//
// Parameter handling is generic: layers expose their parameters and gradients
// as flat slices ([w, b] order), so any optimizer works with any layer
// without per-layer code.  Layers without parameters return empty slices.

pub type Layer = AvgPool2d
	| BatchNorm2d
	| Conv2d
	| Dropout
	| Flatten
	| GlobalAvgPool2d
	| LayerNorm
	| Linear
	| MaxPool2d
	| ReLU
	| Residual
	| Sigmoid
	| Skip
	| Tanh
	| UpSample2d

// forward computes the layer output; layers cache what backward needs.
pub fn (mut l Layer) forward(x mlx.Array) mlx.Array {
	match mut l {
		Linear {
			return l.forward(x)
		}
		Conv2d {
			return l.forward(x)
		}
		ReLU {
			return l.forward(x)
		}
		Sigmoid {
			return l.forward(x)
		}
		Tanh {
			return l.forward(x)
		}
		MaxPool2d {
			return l.forward(x)
		}
		AvgPool2d {
			return l.forward(x)
		}
		GlobalAvgPool2d {
			return l.forward(x)
		}
		UpSample2d {
			return l.forward(x)
		}
		LayerNorm {
			return l.forward(x)
		}
		BatchNorm2d {
			return l.forward(x)
		}
		Dropout {
			return l.forward(x)
		}
		Flatten {
			return l.forward(x)
		}
		Residual {
			return l.forward(x)
		}
		Skip {
			return l.forward(x)
		}
	}
}

// backward takes the gradient w.r.t. the layer output and returns the
// gradient w.r.t. the layer input, recording parameter gradients.
pub fn (mut l Layer) backward(grad mlx.Array) mlx.Array {
	match mut l {
		Linear {
			return l.backward(grad)
		}
		Conv2d {
			return l.backward(grad)
		}
		ReLU {
			return l.backward(grad)
		}
		Sigmoid {
			return l.backward(grad)
		}
		Tanh {
			return l.backward(grad)
		}
		MaxPool2d {
			return l.backward(grad)
		}
		AvgPool2d {
			return l.backward(grad)
		}
		GlobalAvgPool2d {
			return l.backward(grad)
		}
		UpSample2d {
			return l.backward(grad)
		}
		LayerNorm {
			return l.backward(grad)
		}
		BatchNorm2d {
			return l.backward(grad)
		}
		Dropout {
			return l.backward(grad)
		}
		Flatten {
			return l.backward(grad)
		}
		Residual {
			return l.backward(grad)
		}
		Skip {
			return l.backward(grad)
		}
	}
}

// params returns the layer parameters (empty for parameter-free layers).
pub fn (mut l Layer) params() []mlx.Array {
	match mut l {
		Linear {
			return l.params()
		}
		Conv2d {
			return l.params()
		}
		ReLU {
			return l.params()
		}
		Sigmoid {
			return l.params()
		}
		Tanh {
			return l.params()
		}
		MaxPool2d {
			return l.params()
		}
		AvgPool2d {
			return l.params()
		}
		GlobalAvgPool2d {
			return l.params()
		}
		UpSample2d {
			return l.params()
		}
		LayerNorm {
			return l.params()
		}
		BatchNorm2d {
			return l.params()
		}
		Dropout {
			return l.params()
		}
		Flatten {
			return l.params()
		}
		Residual {
			return l.params()
		}
		Skip {
			return l.params()
		}
	}
}

// grads returns the gradients recorded by the last backward call, in the
// same order as params().
pub fn (mut l Layer) grads() []mlx.Array {
	match mut l {
		Linear {
			return l.grads()
		}
		Conv2d {
			return l.grads()
		}
		ReLU {
			return l.grads()
		}
		Sigmoid {
			return l.grads()
		}
		Tanh {
			return l.grads()
		}
		MaxPool2d {
			return l.grads()
		}
		AvgPool2d {
			return l.grads()
		}
		GlobalAvgPool2d {
			return l.grads()
		}
		UpSample2d {
			return l.grads()
		}
		LayerNorm {
			return l.grads()
		}
		BatchNorm2d {
			return l.grads()
		}
		Dropout {
			return l.grads()
		}
		Flatten {
			return l.grads()
		}
		Residual {
			return l.grads()
		}
		Skip {
			return l.grads()
		}
	}
}

// set_params replaces the layer parameters, in the same order as params().
pub fn (mut l Layer) set_params(ps []mlx.Array) {
	match mut l {
		Linear { l.set_params(ps) }
		Conv2d { l.set_params(ps) }
		ReLU { l.set_params(ps) }
		Sigmoid { l.set_params(ps) }
		Tanh { l.set_params(ps) }
		MaxPool2d { l.set_params(ps) }
		AvgPool2d { l.set_params(ps) }
		GlobalAvgPool2d { l.set_params(ps) }
		UpSample2d { l.set_params(ps) }
		LayerNorm { l.set_params(ps) }
		BatchNorm2d { l.set_params(ps) }
		Dropout { l.set_params(ps) }
		Flatten { l.set_params(ps) }
		Residual { l.set_params(ps) }
		Skip { l.set_params(ps) }
	}
}

// set_training switches training-sensitive layers (Dropout, BatchNorm)
// between training and inference behaviour; most layers ignore it.
pub fn (mut l Layer) set_training(training bool) {
	match mut l {
		Linear { l.set_training(training) }
		Conv2d { l.set_training(training) }
		ReLU { l.set_training(training) }
		Sigmoid { l.set_training(training) }
		Tanh { l.set_training(training) }
		MaxPool2d { l.set_training(training) }
		AvgPool2d { l.set_training(training) }
		GlobalAvgPool2d { l.set_training(training) }
		UpSample2d { l.set_training(training) }
		LayerNorm { l.set_training(training) }
		BatchNorm2d { l.set_training(training) }
		Dropout { l.set_training(training) }
		Flatten { l.set_training(training) }
		Residual { l.set_training(training) }
		Skip { l.set_training(training) }
	}
}

// save_params appends the layer parameters to the map under `prefix.*`.
pub fn (mut l Layer) save_params(m mlx.MapStringToArray, prefix string) {
	match mut l {
		Linear { l.save_params(m, prefix) }
		Conv2d { l.save_params(m, prefix) }
		ReLU { l.save_params(m, prefix) }
		Sigmoid { l.save_params(m, prefix) }
		Tanh { l.save_params(m, prefix) }
		MaxPool2d { l.save_params(m, prefix) }
		AvgPool2d { l.save_params(m, prefix) }
		GlobalAvgPool2d { l.save_params(m, prefix) }
		UpSample2d { l.save_params(m, prefix) }
		LayerNorm { l.save_params(m, prefix) }
		BatchNorm2d { l.save_params(m, prefix) }
		Dropout { l.save_params(m, prefix) }
		Flatten { l.save_params(m, prefix) }
		Residual { l.save_params(m, prefix) }
		Skip { l.save_params(m, prefix) }
	}
}

// load_params restores the layer parameters from the map.
pub fn (mut l Layer) load_params(m mlx.MapStringToArray, prefix string) {
	match mut l {
		Linear { l.load_params(m, prefix) }
		Conv2d { l.load_params(m, prefix) }
		ReLU { l.load_params(m, prefix) }
		Sigmoid { l.load_params(m, prefix) }
		Tanh { l.load_params(m, prefix) }
		MaxPool2d { l.load_params(m, prefix) }
		AvgPool2d { l.load_params(m, prefix) }
		GlobalAvgPool2d { l.load_params(m, prefix) }
		UpSample2d { l.load_params(m, prefix) }
		LayerNorm { l.load_params(m, prefix) }
		BatchNorm2d { l.load_params(m, prefix) }
		Dropout { l.load_params(m, prefix) }
		Flatten { l.load_params(m, prefix) }
		Residual { l.load_params(m, prefix) }
		Skip { l.load_params(m, prefix) }
	}
}
