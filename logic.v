// non-OOP — the vjp trampolines must be top-level fns: MLX autograd takes a
// plain C function pointer (mlx.Func) that cannot capture the layer instance.
module nn

import mlx
import mlx_ops

// logic.v — differentiable logic gate networks (LGN).
//
// Each output neuron is wired to two fixed random inputs and computes a
// learnable binary boolean gate.  During training the gate is a softmax over
// the 16 binary gates applied through the multilinear (relaxed) truth table,
// so the layer is fully differentiable; at inference `hard_forward` picks the
// argmax gate per neuron and evaluates exact boolean logic.
//
// Gate indexing: gate g has truth table bits T[g][k] where bit index
// k = 0:(a,b)=(0,0), 1:(0,1), 2:(1,0), 3:(1,1), stored as bit (3-k):
//   AND = 0001 -> 1, OR = 0111 -> 7, XOR = 0110 -> 6, XNOR = 1001 -> 9

// logic_gate_table returns the 16x4 truth-table matrix (flat, row-major).
pub fn logic_gate_table() []f32 {
	mut t := []f32{len: 64}
	for g in 0 .. 16 {
		for k in 0 .. 4 {
			t[g * 4 + k] = f32((g >> (3 - k)) & 1)
		}
	}
	return t
}

// logic_gate_name returns the conventional name of gate g (truth-table index).
pub fn logic_gate_name(g int) string {
	return match g {
		0 { 'FALSE' }
		1 { 'AND' }
		2 { 'a&!b' }
		3 { 'a' }
		4 { '!a&b' }
		5 { 'b' }
		6 { 'XOR' }
		7 { 'OR' }
		8 { 'NOR' }
		9 { 'XNOR' }
		10 { '!b' }
		11 { 'a|!b' }
		12 { '!a' }
		13 { '!a|b' }
		14 { 'NAND' }
		else { 'TRUE' }
	}
}

// logic_table_array returns the truth table as an mlx [16, 4] array.
fn logic_table_array() mlx.Array {
	return mlx.array_f32(logic_gate_table(), [16, 4])
}

// LogicGateLayer is one layer of a differentiable logic gate network.
//
// `residual` appends the layer input to its output (the wire pass-through used
// by differentiable logic gate networks): with only two random wires per gate,
// narrow layers can otherwise lose an entire input's information, leaving the
// network stuck at input-independent constants.  With residual enabled the
// output width is out_dim + in_dim.
pub struct LogicGateLayer {
pub:
	in_dim   int
	out_dim  int
	residual bool
pub mut:
	logits mlx.Array // [out_dim, 16] gate logits
	ix     mlx.Array // [out_dim] int32: first input index per output
	iy     mlx.Array // [out_dim] int32: second input index per output
mut:
	x       mlx.Array // input cached by forward
	dlogits mlx.Array
}

// output_dim returns the width actually produced by forward().
pub fn (l LogicGateLayer) output_dim() int {
	if l.residual {
		return l.out_dim + l.in_dim
	}
	return l.out_dim
}

// new_logic_gate_layer builds a layer with fixed random pairwise wiring and
// small random gate logits.  The logits must NOT be uniform: with all gates
// equally likely the relaxed layer computes the constant 0.5 and the input
// gradient vanishes exactly, so a uniform initialisation deadlocks training.
// `seed` makes the wiring and logits reproducible.
pub fn new_logic_gate_layer(in_dim int, out_dim int, seed u64) LogicGateLayer {
	return new_logic_gate_layer_res(in_dim, out_dim, false, seed)
}

// new_logic_gate_layer_res is new_logic_gate_layer with the residual
// pass-through switch (recommended for hidden layers).
pub fn new_logic_gate_layer_res(in_dim int, out_dim int, residual bool, seed u64) LogicGateLayer {
	return new_logic_gate_layer_wired(in_dim, out_dim, residual, 'random', seed)
}

// new_logic_gate_layer_covered builds a layer with coverage wiring: output i
// reads wire pairs (i % in_dim, (i + 1 + i / in_dim) % in_dim), so every input
// participates and consecutive pairs (including (0,1)) always appear.  This
// matters for narrow nets: with purely random 2-of-N wiring a single output
// gate can miss an entire input, which leaves the network stuck at
// input-independent constants.
pub fn new_logic_gate_layer_covered(in_dim int, out_dim int, residual bool, seed u64) LogicGateLayer {
	return new_logic_gate_layer_wired(in_dim, out_dim, residual, 'coverage', seed)
}

// new_logic_gate_layer_wired builds a layer with a wiring strategy of
// 'random' or 'coverage'.
pub fn new_logic_gate_layer_wired(in_dim int, out_dim int, residual bool, wiring string, seed u64) LogicGateLayer {
	key := mlx_ops.random_key(seed)
	defer {
		key.free()
	}
	mut a := []i32{len: out_dim}
	mut b := []i32{len: out_dim}
	if wiring == 'coverage' {
		for o in 0 .. out_dim {
			a[o] = i32(o % in_dim)
			b[o] = i32((o + 1 + o / in_dim) % in_dim)
		}
	} else {
		ak := mlx_ops.random_randint(mlx.int_scalar(0), mlx.int_scalar(in_dim), [
			out_dim,
		], .int32, key)
		bk := mlx_ops.random_randint(mlx.int_scalar(0), mlx.int_scalar(in_dim), [
			out_dim,
		], .int32, key)
		ad := ak.data_i32()
		bd := bk.data_i32()
		for o in 0 .. out_dim {
			a[o] = i32(ad[o])
			b[o] = i32(bd[o])
		}
	}
	logits := mlx_ops.random_normal([out_dim, 16], .float32, 0.0, 0.5, key)
	return LogicGateLayer{
		in_dim: in_dim
		out_dim: out_dim
		residual: residual
		logits: logits
		ix: mlx.array_i32(a, [out_dim])
		iy: mlx.array_i32(b, [out_dim])
	}
}

// relaxed_basis builds the [n, out, 4] multilinear basis of the two inputs.
fn relaxed_basis(a mlx.Array, b mlx.Array) mlx.Array {
	one := mlx.f32_scalar(1.0)
	na := one.subtract(a)
	nb := one.subtract(b)
	n := a.dim(0)
	o := a.dim(1)
	return mlx.concatenate([na.multiply(nb).reshape([n, o, 1]), na.multiply(b).reshape([
		n,
		o,
		1,
	]), a.multiply(nb).reshape([n, o, 1]), a.multiply(b).reshape([n, o, 1])], 2)
}

// logic_fwd is the autograd trampoline; xs = [x, logits, ix, iy, cfg] with
// cfg an int32 array [residual].  The integer wiring indices carry no
// gradient, so they are detached with stop_gradient (MLX otherwise refuses
// vjp through gather indices).
fn logic_fwd(xs []mlx.Array) []mlx.Array {
	gate := logic_soft(xs[0], xs[1], xs[2].stop_gradient(), xs[3].stop_gradient())
	if xs[4].data_i32()[0] == 1 {
		return [mlx.concatenate([gate, xs[0]], 1)]
	}
	return [gate]
}

pub fn (mut l LogicGateLayer) forward(x mlx.Array) mlx.Array {
	l.x = x
	gate := logic_soft(x, l.logits, l.ix, l.iy)
	if l.residual {
		return mlx.concatenate([gate, x], 1)
	}
	return gate
}

// logic_soft evaluates the relaxed layer (shared by forward and trampoline).
fn logic_soft(x mlx.Array, logits mlx.Array, ix mlx.Array, iy mlx.Array) mlx.Array {
	a := x.take_axis(ix, 1)
	b := x.take_axis(iy, 1)
	basis := relaxed_basis(a, b)
	probs := logits.softmax_axis(1, false)
	m := mlx.einsum('og,gk->ok', [probs, logic_table_array()])
	return mlx.einsum('nok,ok->no', [basis, m])
}

pub fn (mut l LogicGateLayer) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(b2i_logic(l.residual))], [1])
	_, vjps := mlx.vjp(logic_fwd, [l.x, l.logits, l.ix, l.iy, cfg], [grad])
	l.dlogits = vjps[1]
	return vjps[0]
}

// hard_forward evaluates the layer as exact boolean logic: argmax gate per
// neuron, inputs expected in {0, 1} (values > 0.5 count as true).
pub fn (mut l LogicGateLayer) hard_forward(x mlx.Array) mlx.Array {
	idx := l.logits.argmax_axis(1, false) // [out] int32
	rows := logic_table_array().take_axis(idx, 0) // [out, 4]
	a := x.take_axis(l.ix, 1)
	b := x.take_axis(l.iy, 1)
	basis := relaxed_basis(a, b)
	gate := mlx.einsum('nok,ok->no', [basis, rows])
	if l.residual {
		return mlx.concatenate([gate, x], 1)
	}
	return gate
}

// b2i_logic converts the residual flag to a config int.
fn b2i_logic(v bool) int {
	if v {
		return 1
	}
	return 0
}

// gate_probs returns the softmax gate distribution [out_dim, 16].
pub fn (l LogicGateLayer) gate_probs() mlx.Array {
	return l.logits.softmax_axis(1, false)
}

// gate_ids returns the argmax gate index per output neuron.
pub fn (l LogicGateLayer) gate_ids() []int {
	return l.logits.argmax_axis(1, false).data_i32()
}

pub fn (mut l LogicGateLayer) params() []mlx.Array {
	return [l.logits]
}

pub fn (mut l LogicGateLayer) grads() []mlx.Array {
	return [l.dlogits]
}

pub fn (mut l LogicGateLayer) set_params(ps []mlx.Array) {
	l.logits = ps[0]
}

pub fn (mut l LogicGateLayer) set_training(training bool) {}

pub fn (mut l LogicGateLayer) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.logits', l.logits)
	m.insert('${prefix}.ix', l.ix)
	m.insert('${prefix}.iy', l.iy)
}

pub fn (mut l LogicGateLayer) load_params(m mlx.MapStringToArray, prefix string) {
	l.logits = reshape_to(m.get('${prefix}.logits'), [l.out_dim, 16], '${prefix}.logits')
	l.ix = reshape_to(m.get('${prefix}.ix'), [l.out_dim], '${prefix}.ix')
	l.iy = reshape_to(m.get('${prefix}.iy'), [l.out_dim], '${prefix}.iy')
	l.logits.eval()
	l.ix.eval()
	l.iy.eval()
}

// logic_discretized_forward runs a Sequential whose layers are LogicGateLayers
// with argmax gates, i.e. evaluates the learned circuit as exact boolean
// logic.  Non-logic layers are skipped (the network is expected to consist of
// logic layers only).
pub fn logic_discretized_forward(mut net Sequential, x mlx.Array) mlx.Array {
	mut cur := x
	for i in 0 .. net.layers.len {
		mut l := net.layers[i]
		if mut l is LogicGateLayer {
			cur = l.hard_forward(cur)
		}
	}
	return cur
}

// logic_gate_names returns the learned gate names of layer `layer` (empty when
// the layer is not a logic layer or the index is out of range).
pub fn logic_gate_names(net Sequential, layer int) []string {
	if layer < 0 || layer >= net.layers.len {
		return []string{}
	}
	mut l := net.layers[layer]
	if mut l is LogicGateLayer {
		return l.gate_ids().map(logic_gate_name(it))
	}
	return []string{}
}
