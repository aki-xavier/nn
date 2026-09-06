// non-OOP — the vjp trampolines must be top-level fns (see conv2d.v).
module nn

import math
import mlx

// conv1d3d.v — 1D/3D convolution layers with the same vjp-autograd backward
// pattern as Conv2d.  Layouts follow MLX conventions:
//   Conv1d: x [n, l, in],  w [out, k, in]
//   Conv3d: x [n, d, h, w, in], w [out, kd, kh, kw, in]

pub struct Conv1d {
pub:
	in_channels  int
	out_channels int
	kernel_size  int
	stride       int
	padding      int
mut:
	w  mlx.Array
	b  mlx.Array
	x  mlx.Array
	dw mlx.Array
	db mlx.Array
}

pub fn new_conv1d(in_channels int, out_channels int, kernel_size int, stride int, padding int, seed u64) Conv1d {
	fan_in := in_channels * kernel_size
	scale := f32(math.sqrt(2.0 / f64(fan_in)))
	key := mlx.random_key(seed)
	defer {
		key.free()
	}
	return Conv1d{
		in_channels: in_channels
		out_channels: out_channels
		kernel_size: kernel_size
		stride: stride
		padding: padding
		w: mlx.random_normal([out_channels, kernel_size, in_channels], .float32, 0.0, scale, key)
		b: mlx.zeros([1, 1, out_channels], .float32)
	}
}

// conv1d_vjp_fn is the autograd trampoline; xs = [x, w, cfg] with cfg int32
// [stride, padding, 1].
fn conv1d_vjp_fn(xs []mlx.Array) []mlx.Array {
	cfg := xs[2].data_i32()
	return [mlx.conv1d(xs[0], xs[1], cfg[0], cfg[1], cfg[2])]
}

pub fn (mut l Conv1d) forward(x mlx.Array) mlx.Array {
	l.x = x
	return mlx.conv1d(x, l.w, l.stride, l.padding, 1).add(l.b)
}

pub fn (mut l Conv1d) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(l.stride), i32(l.padding), i32(1)], [3])
	_, vjps := mlx.vjp(conv1d_vjp_fn, [l.x, l.w, cfg], [grad])
	l.dw = vjps[1]
	l.db = grad.sum_axes([0, 1], true)
	return vjps[0]
}

pub fn (mut l Conv1d) params() []mlx.Array {
	return [l.w, l.b]
}

pub fn (mut l Conv1d) grads() []mlx.Array {
	return [l.dw, l.db]
}

pub fn (mut l Conv1d) set_params(ps []mlx.Array) {
	l.w = ps[0]
	l.b = ps[1]
}

pub fn (mut l Conv1d) set_training(training bool) {}

pub fn (mut l Conv1d) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w', l.w)
	m.insert('${prefix}.b', l.b)
}

pub fn (mut l Conv1d) load_params(m mlx.MapStringToArray, prefix string) {
	l.w = reshape_to(m.get('${prefix}.w'), [l.out_channels, l.kernel_size, l.in_channels], '${prefix}.w')
	l.b = reshape_to(m.get('${prefix}.b'), [1, 1, l.out_channels], '${prefix}.b')
	l.w.eval()
	l.b.eval()
}

pub struct Conv3d {
pub:
	in_channels  int
	out_channels int
	kernel_size  int
	stride       int
	padding      int
mut:
	w  mlx.Array
	b  mlx.Array
	x  mlx.Array
	dw mlx.Array
	db mlx.Array
}

pub fn new_conv3d(in_channels int, out_channels int, kernel_size int, stride int, padding int, seed u64) Conv3d {
	fan_in := in_channels * kernel_size * kernel_size * kernel_size
	scale := f32(math.sqrt(2.0 / f64(fan_in)))
	key := mlx.random_key(seed)
	defer {
		key.free()
	}
	return Conv3d{
		in_channels: in_channels
		out_channels: out_channels
		kernel_size: kernel_size
		stride: stride
		padding: padding
		w: mlx.random_normal([out_channels, kernel_size, kernel_size, kernel_size, in_channels], .float32, 0.0, scale, key)
		b: mlx.zeros([1, 1, 1, 1, out_channels], .float32)
	}
}

// conv3d_vjp_fn is the autograd trampoline; xs = [x, w, cfg].
fn conv3d_vjp_fn(xs []mlx.Array) []mlx.Array {
	cfg := xs[2].data_i32()
	return [mlx.conv3d(xs[0], xs[1], cfg[0], cfg[1], cfg[2])]
}

pub fn (mut l Conv3d) forward(x mlx.Array) mlx.Array {
	l.x = x
	return mlx.conv3d(x, l.w, l.stride, l.padding, 1).add(l.b)
}

pub fn (mut l Conv3d) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(l.stride), i32(l.padding), i32(1)], [3])
	_, vjps := mlx.vjp(conv3d_vjp_fn, [l.x, l.w, cfg], [grad])
	l.dw = vjps[1]
	l.db = grad.sum_axes([0, 1, 2, 3], true)
	return vjps[0]
}

pub fn (mut l Conv3d) params() []mlx.Array {
	return [l.w, l.b]
}

pub fn (mut l Conv3d) grads() []mlx.Array {
	return [l.dw, l.db]
}

pub fn (mut l Conv3d) set_params(ps []mlx.Array) {
	l.w = ps[0]
	l.b = ps[1]
}

pub fn (mut l Conv3d) set_training(training bool) {}

pub fn (mut l Conv3d) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w', l.w)
	m.insert('${prefix}.b', l.b)
}

pub fn (mut l Conv3d) load_params(m mlx.MapStringToArray, prefix string) {
	l.w = reshape_to(m.get('${prefix}.w'), [l.out_channels, l.kernel_size, l.kernel_size,
		l.kernel_size, l.in_channels], '${prefix}.w')
	l.b = reshape_to(m.get('${prefix}.b'), [1, 1, 1, 1, l.out_channels], '${prefix}.b')
	l.w.eval()
	l.b.eval()
}
