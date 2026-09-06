module nn

import mlx

// layer.v — the Layer sum type unifying every network building block, plus
// the dispatch methods used by Sequential and the optimizers.
//
// NOTE: this V version mis-parses interface method implementations in any
// module that imports mlx, so the framework uses a sum type with match
// dispatch instead of an interface.  New layer types are registered by
// adding them to the sum type below and to each match arm.
//
// Parameter handling is generic: layers expose their parameters and gradients
// as flat slices, so any optimizer works with any layer without per-layer
// code.  Layers without parameters return empty slices.

pub type Layer = AvgPool2d
	| BatchNorm2d
	| CGAGroupLayer
	| CliffordLinear
	| Conv1d
	| Conv2d
	| Conv3d
	| Dropout
	| Flatten
	| GlobalAvgPool2d
	| GRU
	| GroupLayer
	| GroupNorm
	| LSTM
	| LayerNorm
	| Linear
	| MaxPool2d
	| Module
	| MotorGroupLayer
	| ReLU
	| ReprSwitch
	| Residual
	| Sigmoid
	| Skip
	| Tanh
	| UpSample2d
	| Attention

pub fn (mut l Layer) forward(x mlx.Array) mlx.Array {
	match mut l {
		AvgPool2d {
			return l.forward(x)
		}
		BatchNorm2d {
			return l.forward(x)
		}
		CGAGroupLayer {
			return l.forward(x)
		}
		CliffordLinear {
			return l.forward(x)
		}
		Conv1d {
			return l.forward(x)
		}
		Conv2d {
			return l.forward(x)
		}
		Conv3d {
			return l.forward(x)
		}
		Dropout {
			return l.forward(x)
		}
		Flatten {
			return l.forward(x)
		}
		GlobalAvgPool2d {
			return l.forward(x)
		}
		GRU {
			return l.forward(x)
		}
		GroupLayer {
			return l.forward(x)
		}
		GroupNorm {
			return l.forward(x)
		}
		LSTM {
			return l.forward(x)
		}
		LayerNorm {
			return l.forward(x)
		}
		Linear {
			return l.forward(x)
		}
		MaxPool2d {
			return l.forward(x)
		}
		Module {
			return l.forward(x)
		}
		MotorGroupLayer {
			return l.forward(x)
		}
		ReLU {
			return l.forward(x)
		}
		ReprSwitch {
			return l.forward(x)
		}
		Residual {
			return l.forward(x)
		}
		Sigmoid {
			return l.forward(x)
		}
		Skip {
			return l.forward(x)
		}
		Tanh {
			return l.forward(x)
		}
		UpSample2d {
			return l.forward(x)
		}
		Attention {
			return l.forward(x)
		}
	}
}

pub fn (mut l Layer) backward(grad mlx.Array) mlx.Array {
	match mut l {
		AvgPool2d {
			return l.backward(grad)
		}
		BatchNorm2d {
			return l.backward(grad)
		}
		CGAGroupLayer {
			return l.backward(grad)
		}
		CliffordLinear {
			return l.backward(grad)
		}
		Conv1d {
			return l.backward(grad)
		}
		Conv2d {
			return l.backward(grad)
		}
		Conv3d {
			return l.backward(grad)
		}
		Dropout {
			return l.backward(grad)
		}
		Flatten {
			return l.backward(grad)
		}
		GlobalAvgPool2d {
			return l.backward(grad)
		}
		GRU {
			return l.backward(grad)
		}
		GroupLayer {
			return l.backward(grad)
		}
		GroupNorm {
			return l.backward(grad)
		}
		LSTM {
			return l.backward(grad)
		}
		LayerNorm {
			return l.backward(grad)
		}
		Linear {
			return l.backward(grad)
		}
		MaxPool2d {
			return l.backward(grad)
		}
		Module {
			return l.backward(grad)
		}
		MotorGroupLayer {
			return l.backward(grad)
		}
		ReLU {
			return l.backward(grad)
		}
		ReprSwitch {
			return l.backward(grad)
		}
		Residual {
			return l.backward(grad)
		}
		Sigmoid {
			return l.backward(grad)
		}
		Skip {
			return l.backward(grad)
		}
		Tanh {
			return l.backward(grad)
		}
		UpSample2d {
			return l.backward(grad)
		}
		Attention {
			return l.backward(grad)
		}
	}
}

pub fn (mut l Layer) params() []mlx.Array {
	match mut l {
		AvgPool2d {
			return l.params()
		}
		BatchNorm2d {
			return l.params()
		}
		CGAGroupLayer {
			return l.params()
		}
		CliffordLinear {
			return l.params()
		}
		Conv1d {
			return l.params()
		}
		Conv2d {
			return l.params()
		}
		Conv3d {
			return l.params()
		}
		Dropout {
			return l.params()
		}
		Flatten {
			return l.params()
		}
		GlobalAvgPool2d {
			return l.params()
		}
		GRU {
			return l.params()
		}
		GroupLayer {
			return l.params()
		}
		GroupNorm {
			return l.params()
		}
		LSTM {
			return l.params()
		}
		LayerNorm {
			return l.params()
		}
		Linear {
			return l.params()
		}
		MaxPool2d {
			return l.params()
		}
		Module {
			return l.params()
		}
		MotorGroupLayer {
			return l.params()
		}
		ReLU {
			return l.params()
		}
		ReprSwitch {
			return l.params()
		}
		Residual {
			return l.params()
		}
		Sigmoid {
			return l.params()
		}
		Skip {
			return l.params()
		}
		Tanh {
			return l.params()
		}
		UpSample2d {
			return l.params()
		}
		Attention {
			return l.params()
		}
	}
}

pub fn (mut l Layer) grads() []mlx.Array {
	match mut l {
		AvgPool2d {
			return l.grads()
		}
		BatchNorm2d {
			return l.grads()
		}
		CGAGroupLayer {
			return l.grads()
		}
		CliffordLinear {
			return l.grads()
		}
		Conv1d {
			return l.grads()
		}
		Conv2d {
			return l.grads()
		}
		Conv3d {
			return l.grads()
		}
		Dropout {
			return l.grads()
		}
		Flatten {
			return l.grads()
		}
		GlobalAvgPool2d {
			return l.grads()
		}
		GRU {
			return l.grads()
		}
		GroupLayer {
			return l.grads()
		}
		GroupNorm {
			return l.grads()
		}
		LSTM {
			return l.grads()
		}
		LayerNorm {
			return l.grads()
		}
		Linear {
			return l.grads()
		}
		MaxPool2d {
			return l.grads()
		}
		Module {
			return l.grads()
		}
		MotorGroupLayer {
			return l.grads()
		}
		ReLU {
			return l.grads()
		}
		ReprSwitch {
			return l.grads()
		}
		Residual {
			return l.grads()
		}
		Sigmoid {
			return l.grads()
		}
		Skip {
			return l.grads()
		}
		Tanh {
			return l.grads()
		}
		UpSample2d {
			return l.grads()
		}
		Attention {
			return l.grads()
		}
	}
}

pub fn (mut l Layer) set_params(ps []mlx.Array) {
	match mut l {
		AvgPool2d { l.set_params(ps) }
		BatchNorm2d { l.set_params(ps) }
		CGAGroupLayer { l.set_params(ps) }
		CliffordLinear { l.set_params(ps) }
		Conv1d { l.set_params(ps) }
		Conv2d { l.set_params(ps) }
		Conv3d { l.set_params(ps) }
		Dropout { l.set_params(ps) }
		Flatten { l.set_params(ps) }
		GlobalAvgPool2d { l.set_params(ps) }
		GRU { l.set_params(ps) }
		GroupLayer { l.set_params(ps) }
		GroupNorm { l.set_params(ps) }
		LSTM { l.set_params(ps) }
		LayerNorm { l.set_params(ps) }
		Linear { l.set_params(ps) }
		MaxPool2d { l.set_params(ps) }
		Module { l.set_params(ps) }
		MotorGroupLayer { l.set_params(ps) }
		ReLU { l.set_params(ps) }
		ReprSwitch { l.set_params(ps) }
		Residual { l.set_params(ps) }
		Sigmoid { l.set_params(ps) }
		Skip { l.set_params(ps) }
		Tanh { l.set_params(ps) }
		UpSample2d { l.set_params(ps) }
		Attention { l.set_params(ps) }
	}
}

pub fn (mut l Layer) set_training(training bool) {
	match mut l {
		AvgPool2d { l.set_training(training) }
		BatchNorm2d { l.set_training(training) }
		CGAGroupLayer { l.set_training(training) }
		CliffordLinear { l.set_training(training) }
		Conv1d { l.set_training(training) }
		Conv2d { l.set_training(training) }
		Conv3d { l.set_training(training) }
		Dropout { l.set_training(training) }
		Flatten { l.set_training(training) }
		GlobalAvgPool2d { l.set_training(training) }
		GRU { l.set_training(training) }
		GroupLayer { l.set_training(training) }
		GroupNorm { l.set_training(training) }
		LSTM { l.set_training(training) }
		LayerNorm { l.set_training(training) }
		Linear { l.set_training(training) }
		MaxPool2d { l.set_training(training) }
		Module { l.set_training(training) }
		MotorGroupLayer { l.set_training(training) }
		ReLU { l.set_training(training) }
		ReprSwitch { l.set_training(training) }
		Residual { l.set_training(training) }
		Sigmoid { l.set_training(training) }
		Skip { l.set_training(training) }
		Tanh { l.set_training(training) }
		UpSample2d { l.set_training(training) }
		Attention { l.set_training(training) }
	}
}

pub fn (mut l Layer) save_params(m mlx.MapStringToArray, prefix string) {
	match mut l {
		AvgPool2d { l.save_params(m, prefix) }
		BatchNorm2d { l.save_params(m, prefix) }
		CGAGroupLayer { l.save_params(m, prefix) }
		CliffordLinear { l.save_params(m, prefix) }
		Conv1d { l.save_params(m, prefix) }
		Conv2d { l.save_params(m, prefix) }
		Conv3d { l.save_params(m, prefix) }
		Dropout { l.save_params(m, prefix) }
		Flatten { l.save_params(m, prefix) }
		GlobalAvgPool2d { l.save_params(m, prefix) }
		GRU { l.save_params(m, prefix) }
		GroupLayer { l.save_params(m, prefix) }
		GroupNorm { l.save_params(m, prefix) }
		LSTM { l.save_params(m, prefix) }
		LayerNorm { l.save_params(m, prefix) }
		Linear { l.save_params(m, prefix) }
		MaxPool2d { l.save_params(m, prefix) }
		Module { l.save_params(m, prefix) }
		MotorGroupLayer { l.save_params(m, prefix) }
		ReLU { l.save_params(m, prefix) }
		ReprSwitch { l.save_params(m, prefix) }
		Residual { l.save_params(m, prefix) }
		Sigmoid { l.save_params(m, prefix) }
		Skip { l.save_params(m, prefix) }
		Tanh { l.save_params(m, prefix) }
		UpSample2d { l.save_params(m, prefix) }
		Attention { l.save_params(m, prefix) }
	}
}

pub fn (mut l Layer) load_params(m mlx.MapStringToArray, prefix string) {
	match mut l {
		AvgPool2d { l.load_params(m, prefix) }
		BatchNorm2d { l.load_params(m, prefix) }
		CGAGroupLayer { l.load_params(m, prefix) }
		CliffordLinear { l.load_params(m, prefix) }
		Conv1d { l.load_params(m, prefix) }
		Conv2d { l.load_params(m, prefix) }
		Conv3d { l.load_params(m, prefix) }
		Dropout { l.load_params(m, prefix) }
		Flatten { l.load_params(m, prefix) }
		GlobalAvgPool2d { l.load_params(m, prefix) }
		GRU { l.load_params(m, prefix) }
		GroupLayer { l.load_params(m, prefix) }
		GroupNorm { l.load_params(m, prefix) }
		LSTM { l.load_params(m, prefix) }
		LayerNorm { l.load_params(m, prefix) }
		Linear { l.load_params(m, prefix) }
		MaxPool2d { l.load_params(m, prefix) }
		Module { l.load_params(m, prefix) }
		MotorGroupLayer { l.load_params(m, prefix) }
		ReLU { l.load_params(m, prefix) }
		ReprSwitch { l.load_params(m, prefix) }
		Residual { l.load_params(m, prefix) }
		Sigmoid { l.load_params(m, prefix) }
		Skip { l.load_params(m, prefix) }
		Tanh { l.load_params(m, prefix) }
		UpSample2d { l.load_params(m, prefix) }
		Attention { l.load_params(m, prefix) }
	}
}
