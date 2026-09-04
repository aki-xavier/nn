module nn

import math
import mlx

// linear.v — fully connected layer:  y = x·W + b
//
// x has shape [batch, in_features], W [in_features, out_features] and
// b [1, out_features] so it broadcasts over the batch dimension.

pub struct Linear {
pub:
	in_features  int
	out_features int
mut:
	w  mlx.Array
	b  mlx.Array
	x  mlx.Array // input cached by forward for backward
	dw mlx.Array // gradient w.r.t. w, filled by backward
	db mlx.Array // gradient w.r.t. b, filled by backward
}

// new_linear builds a Linear layer with Glorot-style scaled normal weights
// and a zero bias.  `seed` makes initialisation reproducible.
pub fn new_linear(in_features int, out_features int, seed u64) Linear {
	scale := f32(math.sqrt(2.0 / f64(in_features + out_features)))
	key := mlx.random_key(seed)
	defer {
		key.free()
	}
	return Linear{
		in_features: in_features
		out_features: out_features
		w: mlx.random_normal([in_features, out_features], .float32, 0.0, scale, key)
		b: mlx.zeros([1, out_features], .float32)
	}
}

pub fn (mut l Linear) forward(x mlx.Array) mlx.Array {
	l.x = x
	return x.matmul(l.w).add(l.b)
}

pub fn (mut l Linear) backward(grad mlx.Array) mlx.Array {
	l.dw = l.x.transpose().matmul(grad)
	l.db = grad.sum_axis(0, true)
	return grad.matmul(l.w.transpose())
}

pub fn (mut l Linear) params() []mlx.Array {
	return [l.w, l.b]
}

pub fn (mut l Linear) grads() []mlx.Array {
	return [l.dw, l.db]
}

pub fn (mut l Linear) set_params(ps []mlx.Array) {
	l.w = ps[0]
	l.b = ps[1]
}

pub fn (mut l Linear) set_training(training bool) {}

pub fn (mut l Linear) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w', l.w)
	m.insert('${prefix}.b', l.b)
}

pub fn (mut l Linear) load_params(m mlx.MapStringToArray, prefix string) {
	l.w = reshape_to(m.get('${prefix}.w'), [l.in_features, l.out_features], '${prefix}.w (PyTorch linear weights need perm [1, 0])')
	l.b = reshape_to(m.get('${prefix}.b'), [1, l.out_features], '${prefix}.b')
	l.w.eval()
	l.b.eval()
}
