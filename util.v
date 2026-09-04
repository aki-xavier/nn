// non-OOP — one tiny shape helper shared by layer load_params implementations.
module nn

import mlx

// util.v — small shared helpers.

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
