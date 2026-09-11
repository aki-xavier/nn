// non-OOP — test file: free test_/helper functions are exempt from the OOP rule.
module nn

import mlx

// logic_test.v — differentiable logic gate layer: truth-table sanity, exact
// boolean (discretised) inference, finite-difference gradients, and learning
// XOR/AND followed by exact discretisation.

fn test_logic_gate_table() {
	t := logic_gate_table()
	// AND = gate 1: bits (0,0)=0,(0,1)=0,(1,0)=0,(1,1)=1
	assert t[1 * 4 + 0] == 0.0 && t[1 * 4 + 3] == 1.0
	// OR = gate 7
	assert t[7 * 4 + 1] == 1.0 && t[7 * 4 + 0] == 0.0
	// XOR = gate 6, XNOR = gate 9
	assert t[6 * 4 + 1] == 1.0 && t[6 * 4 + 2] == 1.0 && t[6 * 4 + 0] == 0.0 && t[6 * 4 + 3] == 0.0
	assert t[9 * 4 + 0] == 1.0 && t[9 * 4 + 3] == 1.0 && t[9 * 4 + 1] == 0.0
	assert logic_gate_name(1) == 'AND' && logic_gate_name(6) == 'XOR'
}

// set_gate forces every neuron of the layer to gate g (near-one-hot logits).
fn set_gate(mut l LogicGateLayer, g int) {
	mut logits := []f32{len: l.out_dim * 16, init: f32(-8.0)}
	for o in 0 .. l.out_dim {
		logits[o * 16 + g] = 8.0
	}
	l.set_params([mlx.array_f32(logits, [l.out_dim, 16])])
}

fn test_logic_hard_forward_exact() {
	// a single gate wired to inputs (0,1) forced to AND/OR/XOR/XNOR and
	// evaluated on all four boolean combinations
	x := mlx.array_f32([f32(0), 0, 0, 1, 1, 0, 1, 1], [4, 2])
	for spec in [[1, 0, 0, 0, 1], [7, 0, 1, 1, 1], [6, 0, 1, 1, 0], [9, 1, 0, 0, 1]] {
		mut l := LogicGateLayer{
			in_dim: 2
			out_dim: 1
			logits: mlx.zeros([1, 16], .float32)
			ix: mlx.array_i32([i32(0)], [1])
			iy: mlx.array_i32([i32(1)], [1])
		}
		set_gate(mut l, spec[0])
		got := l.hard_forward(x).data_f32()
		assert got[0] == f32(spec[1]) && got[1] == f32(spec[2]) && got[2] == f32(spec[3]) && got[3] == f32(spec[4]), 'gate ${spec[0]} (${logic_gate_name(spec[0])}): got ${got}'
	}
}

fn test_logic_gradient() {
	mut l := Layer(new_logic_gate_layer_covered(6, 5, true, 7))
	x := mlx.array_f32([]f32{len: 3 * 6, init: f32((index * 7) % 5) / 4.0}, [3, 6])
	fd_check('logic.logits', mut l, x, 0)
}

// build_covered_net wires a three-layer covered/residual gate network.
fn build_covered_net(n_in int, width int, seed u64) Sequential {
	mut net := Sequential{}
	net.add(new_logic_gate_layer_covered(n_in, width, true, seed))
	net.add(new_logic_gate_layer_covered(n_in + width, width, true, seed + 1))
	net.add(new_logic_gate_layer_covered(n_in + 2 * width, 1, false, seed + 2))
	return net
}

// bool_table builds the 2^n input table (LSB-first bit order).
fn bool_table(n_in int) mlx.Array {
	n := 1 << n_in
	mut xs := []f32{}
	for i in 0 .. n {
		for b in 0 .. n_in {
			xs << f32((i >> b) & 1)
		}
	}
	return mlx.array_f32(xs, [n, n_in])
}

// discretized_forward runs the trained net with argmax gates (exact boolean).
fn discretized_forward(mut net Sequential, x mlx.Array) []f32 {
	mut cur := x
	for i in 0 .. net.layers.len {
		mut l := net.layers[i]
		if mut l is LogicGateLayer {
			cur = l.hard_forward(cur)
		}
	}
	return cur.data_f32()
}

fn test_logic_learns_xor_and_discretizes() {
	n_in := 2
	x := bool_table(n_in)
	y := mlx.array_f32([f32(0), 1, 1, 0], [4, 1]) // XOR
	mut net := build_covered_net(n_in, 8, 41)
	mut dl := new_dataloader(Dataset{
		x: x
		y: y
	}, 4, true)
	mut mse := Loss(MSELoss{})
	mut opt := Optimizer(Adam{
		lr: 0.05
	})
	net.fit_loader(mut dl, mut mse, mut opt, 800, 0)
	net.set_training(false)
	soft := net.predict(x)
	soft_err := soft.subtract(y).abs().max().item_f32()
	assert soft_err < 0.05, 'relaxed XOR not learned: err ${soft_err}'

	hard := discretized_forward(mut net, x)
	want := y.data_f32()
	for i in 0 .. 4 {
		assert absf(hard[i] - want[i]) < 1e-5, 'discretised XOR wrong at ${i}: ${hard} vs ${want}'
	}
}

fn test_logic_learns_and() {
	n_in := 2
	x := bool_table(n_in)
	y := mlx.array_f32([f32(0), 0, 0, 1], [4, 1]) // AND
	mut net := build_covered_net(n_in, 8, 51)
	mut dl := new_dataloader(Dataset{
		x: x
		y: y
	}, 4, true)
	mut mse := Loss(MSELoss{})
	mut opt := Optimizer(Adam{
		lr: 0.05
	})
	net.fit_loader(mut dl, mut mse, mut opt, 800, 0)
	net.set_training(false)
	err := net.predict(x).subtract(y).abs().max().item_f32()
	assert err < 0.05, 'relaxed AND not learned: err ${err}'
	hard := discretized_forward(mut net, x)
	want := y.data_f32()
	for i in 0 .. 4 {
		assert absf(hard[i] - want[i]) < 1e-5, 'discretised AND wrong at ${i}: ${hard} vs ${want}'
	}
}
