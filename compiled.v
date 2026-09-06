// non-OOP — net_forward_arrays must be a top-level fn: MLX compiles plain C
// function pointer callbacks that cannot capture values, so the dispatcher
// travels through the mlx PayloadPair payload slot instead.
module nn

import mlx

// compiled.v — Sequential.compile(): MLX graph compilation for inference
// acceleration.
//
// mlx.Func cannot capture the network instance, so the same trampoline trick
// as the vjp layers applies, with one upgrade: the mlx payload slot
// (PayloadPair) carries the network pointer, which lets the compiled graph
// run the **real** layer forward functions instead of re-implementing layer
// math in the trampoline.  The compiled closure is produced by mlx.compile
// over that payload closure.
//
// Lifetime: keep the CompiledNet (and the network) alive while applying it;
// the compiled closure references the network through the payload.

// CompiledNet is an mlx-compiled view of a Sequential network.
pub struct CompiledNet {
mut:
	net      &Sequential
	pair     &mlx.PayloadPair
	compiled mlx.Closure
}

// net_forward_arrays is the compiled dispatcher: runs the payload net's
// forward on the first input array.
fn net_forward_arrays(xs []mlx.Array, data voidptr) []mlx.Array {
	mut net := unsafe { &Sequential(data) }
	return [net.forward(xs[0])]
}

// compile returns an mlx-compiled view of the network for fast inference.
// The returned CompiledNet keeps the network, the payload pair and the
// compiled closure alive.
pub fn (mut net Sequential) compile() CompiledNet {
	unsafe {
		pair := &mlx.PayloadPair{
			f: net_forward_arrays
			data: voidptr(&net)
		}
		base := mlx.new_payload_closure(pair)
		compiled := base.compile_closure(false)
		return CompiledNet{
			net: &net
			pair: pair
			compiled: compiled
		}
	}
}

// apply runs the compiled graph on x (single input).
pub fn (mut c CompiledNet) apply(x mlx.Array) mlx.Array {
	outs := c.compiled.apply([x])
	return outs[0]
}
