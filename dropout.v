module nn

import mlx

// dropout.v — inverted dropout: at training time each element is zeroed with
// probability p and survivors are scaled by 1/(1-p); at inference time the
// layer is the identity.

pub struct Dropout {
pub:
	p f32
mut:
	mask     mlx.Array // keep mask cached by forward
	training bool = true
}

pub fn new_dropout(p f32) Dropout {
	return Dropout{
		p: p
	}
}

pub fn (mut l Dropout) forward(x mlx.Array) mlx.Array {
	if !l.training || l.p <= 0.0 {
		l.mask = mlx.empty()
		return x
	}
	keep := mlx.f32_scalar(1.0 - l.p)
	key := mlx.no_key()
	l.mask = mlx.random_bernoulli(keep, x.shape(), key)
	return mlx.s_div(x.multiply(l.mask), f64(1.0 - l.p))
}

pub fn (mut l Dropout) backward(grad mlx.Array) mlx.Array {
	if !l.training || l.p <= 0.0 {
		return grad
	}
	return mlx.s_div(grad.multiply(l.mask), f64(1.0 - l.p))
}

pub fn (mut l Dropout) params() []mlx.Array {
	return []
}

pub fn (mut l Dropout) grads() []mlx.Array {
	return []
}

pub fn (mut l Dropout) set_params(ps []mlx.Array) {}

pub fn (mut l Dropout) set_training(training bool) {
	l.training = training
}

pub fn (mut l Dropout) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut l Dropout) load_params(m mlx.MapStringToArray, prefix string) {}
