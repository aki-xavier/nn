module nn

import mlx

// pool.v — pooling and upsampling layers for NHWC tensors.
//
// MaxPool2d/AvgPool2d are non-overlapping (kernel == stride) and require the
// spatial dims to be divisible by the kernel; UpSample2d does nearest
// neighbour scaling by an integer factor.  All are built from
// reshape/broadcast/reduce so the backward passes stay simple.

// grad_upsample spreads a reduced gradient [n, h/k, w/k, c] back to the full
// [n, h, w, c] shape, replicating each value over its k×k window.
fn grad_upsample(g mlx.Array, shape []int, k int) mlx.Array {
	n := shape[0]
	h := shape[1]
	w := shape[2]
	c := shape[3]
	return g.reshape([n, h / k, 1, w / k, 1, c]).broadcast_to([n, h / k, k, w / k, k, c]).reshape([
		n,
		h,
		w,
		c,
	])
}

// check_pool_shape panics unless x is NHWC with spatial dims divisible by k.
fn check_pool_shape(x mlx.Array, k int) []int {
	shape := x.shape()
	if shape.len != 4 {
		panic('nn: pooling layers expect NHWC input, got ${shape}')
	}
	if shape[1] % k != 0 || shape[2] % k != 0 {
		panic('nn: pooling input ${shape} not divisible by kernel ${k}')
	}
	return shape
}

// MaxPool2d pools each k×k window to its maximum.
pub struct MaxPool2d {
pub:
	kernel int
mut:
	x mlx.Array // input cached by forward
}

pub fn new_max_pool2d(kernel int) MaxPool2d {
	return MaxPool2d{
		kernel: kernel
	}
}

pub fn (mut l MaxPool2d) forward(x mlx.Array) mlx.Array {
	l.x = x
	shape := check_pool_shape(x, l.kernel)
	k := l.kernel
	r := x.reshape([shape[0], shape[1] / k, k, shape[2] / k, k, shape[3]])
	return r.max_axes([2, 4], false)
}

pub fn (mut l MaxPool2d) backward(grad mlx.Array) mlx.Array {
	shape := l.x.shape()
	k := l.kernel
	x6 := l.x.reshape([shape[0], shape[1] / k, k, shape[2] / k, k, shape[3]])
	p := x6.max_axes([2, 4], true).broadcast_to(x6.shape())
	// 1 at the window maximum, fractional on ties so gradient mass is conserved.
	mask := mlx.where(x6.equal(p), mlx.ones_like(x6), mlx.zeros_like(x6))
	ties := mask.sum_axes([2, 4], true).broadcast_to(x6.shape())
	g6 := grad.reshape([shape[0], shape[1] / k, 1, shape[2] / k, 1, shape[3]]).broadcast_to(x6.shape())
	return g6.multiply(mask).divide(ties).reshape(shape)
}

pub fn (mut l MaxPool2d) params() []mlx.Array {
	return []
}

pub fn (mut l MaxPool2d) grads() []mlx.Array {
	return []
}

pub fn (mut l MaxPool2d) set_params(ps []mlx.Array) {}

pub fn (mut l MaxPool2d) set_training(training bool) {}

pub fn (mut l MaxPool2d) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut l MaxPool2d) load_params(m mlx.MapStringToArray, prefix string) {}

// AvgPool2d pools each k×k window to its mean.
pub struct AvgPool2d {
pub:
	kernel int
mut:
	shape []int // input shape cached by forward
}

pub fn new_avg_pool2d(kernel int) AvgPool2d {
	return AvgPool2d{
		kernel: kernel
	}
}

pub fn (mut l AvgPool2d) forward(x mlx.Array) mlx.Array {
	l.shape = check_pool_shape(x, l.kernel)
	k := l.kernel
	r := x.reshape([l.shape[0], l.shape[1] / k, k, l.shape[2] / k, k, l.shape[3]])
	return r.mean_axes([2, 4], false)
}

pub fn (mut l AvgPool2d) backward(grad mlx.Array) mlx.Array {
	return mlx.s_mul(grad_upsample(grad, l.shape, l.kernel), 1.0 / f64(l.kernel * l.kernel))
}

pub fn (mut l AvgPool2d) params() []mlx.Array {
	return []
}

pub fn (mut l AvgPool2d) grads() []mlx.Array {
	return []
}

pub fn (mut l AvgPool2d) set_params(ps []mlx.Array) {}

pub fn (mut l AvgPool2d) set_training(training bool) {}

pub fn (mut l AvgPool2d) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut l AvgPool2d) load_params(m mlx.MapStringToArray, prefix string) {}

// GlobalAvgPool2d averages each channel over the whole spatial extent,
// producing [n, 1, 1, c].
pub struct GlobalAvgPool2d {
mut:
	shape []int // input shape cached by forward
}

pub fn (mut l GlobalAvgPool2d) forward(x mlx.Array) mlx.Array {
	l.shape = check_pool_shape(x, 1)
	return x.mean_axes([1, 2], true)
}

pub fn (mut l GlobalAvgPool2d) backward(grad mlx.Array) mlx.Array {
	hw := f64(l.shape[1] * l.shape[2])
	return mlx.s_mul(grad.broadcast_to(l.shape), 1.0 / hw)
}

pub fn (mut l GlobalAvgPool2d) params() []mlx.Array {
	return []
}

pub fn (mut l GlobalAvgPool2d) grads() []mlx.Array {
	return []
}

pub fn (mut l GlobalAvgPool2d) set_params(ps []mlx.Array) {}

pub fn (mut l GlobalAvgPool2d) set_training(training bool) {}

pub fn (mut l GlobalAvgPool2d) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut l GlobalAvgPool2d) load_params(m mlx.MapStringToArray, prefix string) {}

// UpSample2d scales spatial dims by an integer factor (nearest neighbour).
pub struct UpSample2d {
pub:
	scale int
mut:
	shape []int // input shape cached by forward
}

pub fn new_upsample2d(scale int) UpSample2d {
	return UpSample2d{
		scale: scale
	}
}

pub fn (mut l UpSample2d) forward(x mlx.Array) mlx.Array {
	l.shape = check_pool_shape(x, 1)
	s := l.scale
	return x.reshape([l.shape[0], l.shape[1], 1, l.shape[2], 1, l.shape[3]]).broadcast_to([
		l.shape[0],
		l.shape[1],
		s,
		l.shape[2],
		s,
		l.shape[3],
	]).reshape([l.shape[0], l.shape[1] * s, l.shape[2] * s, l.shape[3]])
}

pub fn (mut l UpSample2d) backward(grad mlx.Array) mlx.Array {
	s := l.scale
	g6 := grad.reshape([l.shape[0], l.shape[1], s, l.shape[2], s, l.shape[3]])
	return g6.sum_axes([2, 4], false)
}

pub fn (mut l UpSample2d) params() []mlx.Array {
	return []
}

pub fn (mut l UpSample2d) grads() []mlx.Array {
	return []
}

pub fn (mut l UpSample2d) set_params(ps []mlx.Array) {}

pub fn (mut l UpSample2d) set_training(training bool) {}

pub fn (mut l UpSample2d) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut l UpSample2d) load_params(m mlx.MapStringToArray, prefix string) {}
