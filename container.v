module nn

import mlx

// container.v — structural layers: shape glue and skip-connection containers.
//
// Residual and Skip hold their own inner layer stack, giving the framework
// branching topologies (ResNet-style addition, U-Net-style channel concat)
// while Sequential stays a linear chain.  Parameter/gradient/save/load
// methods recurse into the inner stack, so optimizers and persistence work
// unchanged.

// Flatten reshapes [n, ...] into [n, prod(rest)].
pub struct Flatten {
mut:
	shape []int // input shape cached by forward
}

pub fn (mut l Flatten) forward(x mlx.Array) mlx.Array {
	l.shape = x.shape()
	mut rest := 1
	for d in l.shape[1..] {
		rest *= d
	}
	return x.reshape([l.shape[0], rest])
}

pub fn (mut l Flatten) backward(grad mlx.Array) mlx.Array {
	return grad.reshape(l.shape)
}

pub fn (mut l Flatten) params() []mlx.Array {
	return []
}

pub fn (mut l Flatten) grads() []mlx.Array {
	return []
}

pub fn (mut l Flatten) set_params(ps []mlx.Array) {}

pub fn (mut l Flatten) set_training(training bool) {}

pub fn (mut l Flatten) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut l Flatten) load_params(m mlx.MapStringToArray, prefix string) {}

// Residual computes x + inner(x) (ResNet-style skip by addition).
pub struct Residual {
mut:
	layers []Layer
}

pub fn new_residual(layers []Layer) Residual {
	return Residual{
		layers: layers
	}
}

pub fn (mut l Residual) forward(x mlx.Array) mlx.Array {
	mut out := x
	for mut il in l.layers {
		out = il.forward(out)
	}
	return x.add(out)
}

pub fn (mut l Residual) backward(grad mlx.Array) mlx.Array {
	mut g := grad
	for i := l.layers.len - 1; i >= 0; i-- {
		g = l.layers[i].backward(g)
	}
	return grad.add(g)
}

pub fn (mut l Residual) params() []mlx.Array {
	mut out := []mlx.Array{}
	for mut il in l.layers {
		out << il.params()
	}
	return out
}

pub fn (mut l Residual) grads() []mlx.Array {
	mut out := []mlx.Array{}
	for mut il in l.layers {
		out << il.grads()
	}
	return out
}

pub fn (mut l Residual) set_params(ps []mlx.Array) {
	mut off := 0
	for mut il in l.layers {
		n := il.params().len
		if n > 0 {
			il.set_params(ps[off..off + n])
		}
		off += n
	}
}

pub fn (mut l Residual) set_training(training bool) {
	for mut il in l.layers {
		il.set_training(training)
	}
}

pub fn (mut l Residual) save_params(m mlx.MapStringToArray, prefix string) {
	for i, mut il in l.layers {
		il.save_params(m, '${prefix}.inner.${i}')
	}
}

pub fn (mut l Residual) load_params(m mlx.MapStringToArray, prefix string) {
	for i, mut il in l.layers {
		il.load_params(m, '${prefix}.inner.${i}')
	}
}

// Skip computes concatenate([x, inner(x)]) along the channel (last) axis —
// the U-Net decoder pattern.
pub struct Skip {
mut:
	layers []Layer
	in_ch  int // input channel count cached by forward
}

pub fn new_skip(layers []Layer) Skip {
	return Skip{
		layers: layers
	}
}

pub fn (mut l Skip) forward(x mlx.Array) mlx.Array {
	shape := x.shape()
	l.in_ch = shape[shape.len - 1]
	mut out := x
	for mut il in l.layers {
		out = il.forward(out)
	}
	return mlx.concatenate([x, out], -1)
}

pub fn (mut l Skip) backward(grad mlx.Array) mlx.Array {
	shape := grad.shape()
	last := shape[shape.len - 1]
	in_ch := l.in_ch
	lo_idx := mlx.array_i32([]int{len: in_ch, init: index}.map(i32(it)), [in_ch])
	hi_idx := mlx.array_i32([]int{len: last - in_ch, init: in_ch + index}.map(i32(it)), [
		last - in_ch,
	])
	gx := grad.take_axis(lo_idx, -1)
	mut g := grad.take_axis(hi_idx, -1)
	for i := l.layers.len - 1; i >= 0; i-- {
		g = l.layers[i].backward(g)
	}
	return gx.add(g)
}

pub fn (mut l Skip) params() []mlx.Array {
	mut out := []mlx.Array{}
	for mut il in l.layers {
		out << il.params()
	}
	return out
}

pub fn (mut l Skip) grads() []mlx.Array {
	mut out := []mlx.Array{}
	for mut il in l.layers {
		out << il.grads()
	}
	return out
}

pub fn (mut l Skip) set_params(ps []mlx.Array) {
	mut off := 0
	for mut il in l.layers {
		n := il.params().len
		if n > 0 {
			il.set_params(ps[off..off + n])
		}
		off += n
	}
}

pub fn (mut l Skip) set_training(training bool) {
	for mut il in l.layers {
		il.set_training(training)
	}
}

pub fn (mut l Skip) save_params(m mlx.MapStringToArray, prefix string) {
	for i, mut il in l.layers {
		il.save_params(m, '${prefix}.inner.${i}')
	}
}

pub fn (mut l Skip) load_params(m mlx.MapStringToArray, prefix string) {
	for i, mut il in l.layers {
		il.load_params(m, '${prefix}.inner.${i}')
	}
}
