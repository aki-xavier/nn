// non-OOP — one tiny shape helper shared by layer load_params implementations,
// plus the finite-difference gradient checker used by the test suites.
module nn

import mlx

// util.v — small shared helpers.

fn absf(v f32) f32 {
	if v < 0 {
		return -v
	}
	return v
}

// reshape_to returns t reshaped to `want` when the element counts match
// (handles 1-D PyTorch biases vs our broadcast-shaped biases), and panics
// with a descriptive message otherwise.
fn reshape_to(t mlx.Array, want []int, key string) mlx.Array {
	if t.shape() == want {
		return t
	}
	mut n := 1
	for d in want {
		n *= d
	}
	if int(t.size()) == n {
		return t.reshape(want)
	}
	panic('nn: ${key} shape ${t.shape()} does not match ${want}')
	return t
}

// set_param_f32 replaces parameter idx of layer l with flat f32 values.
fn set_param_f32(mut l Layer, idx int, vals []f32) {
	mut ps := l.params()
	shape := ps[idx].shape()
	ps[idx] = mlx.array_f32(vals, shape)
	l.set_params(ps)
}

// fd_check compares analytic backward gradients with finite differences on
// parameter idx, for an all-ones cotangent.
fn fd_check(name string, mut l Layer, x mlx.Array, idx int) {
	out := l.forward(x)
	g := mlx.ones_like(out)
	l.backward(g)
	analytic := l.grads()[idx].data_f32()

	base := l.params()[idx].data_f32()
	eps := f32(1e-3)
	mut bad := 0
	mut ncheck := analytic.len
	if ncheck > 8 {
		ncheck = 8
	}
	for i in 0 .. ncheck {
		mut plus := base.clone()
		plus[i] = base[i] + eps
		mut minus := base.clone()
		minus[i] = base[i] - eps
		set_param_f32(mut l, idx, plus)
		hi := l.forward(x).sum().item_f32()
		set_param_f32(mut l, idx, minus)
		lo := l.forward(x).sum().item_f32()
		numeric := (hi - lo) / (2 * eps)
		if absf(analytic[i] - numeric) > 0.05 * (absf(numeric) + 1.0) {
			bad++
			eprintln('${name} param ${idx} elem ${i}: analytic=${analytic[i]:.5f} numeric=${numeric:.5f}')
		}
	}
	set_param_f32(mut l, idx, base)
	assert bad == 0, '${name}: ${bad} mismatched gradient elements'
}
