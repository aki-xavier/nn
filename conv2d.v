// non-OOP — conv2d_vjp_fn must be a top-level fn: MLX autograd takes a plain
// C function pointer (mlx.Func) that cannot capture the layer instance.
module nn

import math
import mlx

// conv2d.v — 2D convolution layer (NHWC input, weight [out, k, k, in]).
//
// The backward pass uses MLX autograd (vjp) instead of hand-written gradient
// formulas: conv gradients decompose into transposed/ordinary convolutions
// with stride/padding bookkeeping that is easy to get wrong by hand.

pub struct Conv2d {
pub:
	in_channels  int
	out_channels int
	kernel_size  int
	stride       int
	padding      int
mut:
	w  mlx.Array // [out_channels, k, k, in_channels]
	b  mlx.Array // [1, 1, 1, out_channels]
	x  mlx.Array // input cached by forward
	dw mlx.Array
	db mlx.Array
}

// new_conv2d builds a Conv2d layer with He-style scaled normal weights and a
// zero bias.  `seed` makes initialisation reproducible.
pub fn new_conv2d(in_channels int, out_channels int, kernel_size int, stride int, padding int, seed u64) Conv2d {
	fan_in := in_channels * kernel_size * kernel_size
	scale := f32(math.sqrt(2.0 / f64(fan_in)))
	key := mlx.random_key(seed)
	defer {
		key.free()
	}
	return Conv2d{
		in_channels: in_channels
		out_channels: out_channels
		kernel_size: kernel_size
		stride: stride
		padding: padding
		w: mlx.random_normal([out_channels, kernel_size, kernel_size, in_channels], .float32, 0.0, scale, key)
		b: mlx.zeros([1, 1, 1, out_channels], .float32)
	}
}

// conv2d_vjp_fn is the autograd trampoline for Conv2d.backward.  xs carries
// [x, w, cfg] where cfg is an int32 array [stride, padding, groups]; the
// config travels as an array because the trampoline cannot capture state.
fn conv2d_vjp_fn(xs []mlx.Array) []mlx.Array {
	cfg := xs[2].data_i32()
	return [mlx.conv2d(xs[0], xs[1], cfg[0], cfg[1], cfg[2])]
}

pub fn (mut l Conv2d) forward(x mlx.Array) mlx.Array {
	l.x = x
	return mlx.conv2d(x, l.w, l.stride, l.padding, 1).add(l.b)
}

pub fn (mut l Conv2d) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(l.stride), i32(l.padding), i32(1)], [3])
	_, vjps := mlx.vjp(conv2d_vjp_fn, [l.x, l.w, cfg], [grad])
	l.dw = vjps[1]
	l.db = grad.sum_axes([0, 1, 2], true)
	return vjps[0]
}

pub fn (mut l Conv2d) params() []mlx.Array {
	return [l.w, l.b]
}

pub fn (mut l Conv2d) grads() []mlx.Array {
	return [l.dw, l.db]
}

pub fn (mut l Conv2d) set_params(ps []mlx.Array) {
	l.w = ps[0]
	l.b = ps[1]
}

pub fn (mut l Conv2d) set_training(training bool) {}

pub fn (mut l Conv2d) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w', l.w)
	m.insert('${prefix}.b', l.b)
}

pub fn (mut l Conv2d) load_params(m mlx.MapStringToArray, prefix string) {
	l.w = m.get('${prefix}.w')
	l.b = m.get('${prefix}.b')
	// safetensors arrays are lazy Load primitives bound to the CPU stream;
	// materialise them so later GPU ops read real data.
	l.w.eval()
	l.b.eval()
}
