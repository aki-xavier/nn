module nn

import mlx

// activations.v — stateless activation layers.  They cache their input or
// output during forward so backward can apply the local derivative.

// ReLU clamps negative inputs to zero:  f(x) = max(0, x)
pub struct ReLU {
mut:
	x mlx.Array // input cached by forward
}

pub fn (mut a ReLU) forward(x mlx.Array) mlx.Array {
	a.x = x
	return x.maximum(mlx.f32_scalar(0.0))
}

pub fn (mut a ReLU) backward(grad mlx.Array) mlx.Array {
	mask := mlx.where(mlx.s_gt(a.x, 0.0), mlx.ones_like(a.x), mlx.zeros_like(a.x))
	return grad.multiply(mask)
}

pub fn (mut a ReLU) params() []mlx.Array {
	return []
}

pub fn (mut a ReLU) grads() []mlx.Array {
	return []
}

pub fn (mut a ReLU) set_params(ps []mlx.Array) {}

pub fn (mut a ReLU) set_training(training bool) {}

pub fn (mut a ReLU) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut a ReLU) load_params(m mlx.MapStringToArray, prefix string) {}

// Sigmoid squashes to (0, 1):  f(x) = 1 / (1 + e^-x)
pub struct Sigmoid {
mut:
	out mlx.Array // output cached by forward
}

pub fn (mut a Sigmoid) forward(x mlx.Array) mlx.Array {
	a.out = x.sigmoid()
	return a.out
}

pub fn (mut a Sigmoid) backward(grad mlx.Array) mlx.Array {
	// f'(x) = f(x) · (1 - f(x))
	return grad.multiply(a.out.multiply(mlx.s_rsub(a.out, 1.0)))
}

pub fn (mut a Sigmoid) params() []mlx.Array {
	return []
}

pub fn (mut a Sigmoid) grads() []mlx.Array {
	return []
}

pub fn (mut a Sigmoid) set_params(ps []mlx.Array) {}

pub fn (mut a Sigmoid) set_training(training bool) {}

pub fn (mut a Sigmoid) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut a Sigmoid) load_params(m mlx.MapStringToArray, prefix string) {}

// Tanh squashes to (-1, 1):  f(x) = tanh(x)
pub struct Tanh {
mut:
	out mlx.Array // output cached by forward
}

pub fn (mut a Tanh) forward(x mlx.Array) mlx.Array {
	a.out = x.tanh()
	return a.out
}

pub fn (mut a Tanh) backward(grad mlx.Array) mlx.Array {
	// f'(x) = 1 - f(x)²
	return grad.multiply(mlx.s_rsub(a.out.square(), 1.0))
}

pub fn (mut a Tanh) params() []mlx.Array {
	return []
}

pub fn (mut a Tanh) grads() []mlx.Array {
	return []
}

pub fn (mut a Tanh) set_params(ps []mlx.Array) {}

pub fn (mut a Tanh) set_training(training bool) {}

pub fn (mut a Tanh) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut a Tanh) load_params(m mlx.MapStringToArray, prefix string) {}
