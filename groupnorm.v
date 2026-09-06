module nn

import mlx

// GroupNorm normalises channels in `groups` groups, over the spatial axes
// plus each group's channels, with per-channel affine parameters.  No running
// statistics: training and inference use the same batch statistics.
pub struct GroupNorm {
pub:
	channels int
	groups   int
	eps      f32 = 1e-5
mut:
	w  mlx.Array // gamma [c]
	b  mlx.Array // beta [c]
	x  mlx.Array
	dw mlx.Array
	db mlx.Array
}

pub fn new_group_norm(channels int, groups int) GroupNorm {
	return GroupNorm{
		channels: channels
		groups: groups
		w: mlx.ones([channels], .float32)
		b: mlx.zeros([channels], .float32)
	}
}

// gn_vjp_fn is the autograd trampoline; xs = [x, gamma, beta, cfg] with cfg
// int32 [channels, groups], eps carried as a float array.
fn gn_vjp_fn(xs []mlx.Array) []mlx.Array {
	cfg := xs[3].data_i32()
	c := cfg[0]
	g := cfg[1]
	eps := xs[4].item_f32()
	shape := xs[0].shape()
	cpg := c / g
	// x [n, .., c] -> [n, .., g, cpg]; normalise over all but the last two
	// group axes and the channel-within-group axis.
	xg := xs[0].reshape([shape[0], -1, g, cpg])
	mu := xg.mean_axes([1, 3], true)
	xc := xg.subtract(mu)
	var_ := xc.square().mean_axes([1, 3], true)
	yn := xc.divide(var_.add(mlx.f32_scalar(eps)).sqrt())
	y := yn.multiply(xs[1].reshape([1, 1, g, cpg])).add(xs[2].reshape([1, 1, g, cpg]))
	return [y.reshape(shape)]
}

pub fn (mut l GroupNorm) forward(x mlx.Array) mlx.Array {
	l.x = x
	shape := x.shape()
	cpg := l.channels / l.groups
	xg := x.reshape([shape[0], -1, l.groups, cpg])
	mu := xg.mean_axes([1, 3], true)
	xc := xg.subtract(mu)
	var_ := xc.square().mean_axes([1, 3], true)
	yn := xc.divide(var_.add(mlx.f32_scalar(l.eps)).sqrt())
	y := yn.multiply(l.w.reshape([1, 1, l.groups, cpg])).add(l.b.reshape([1, 1, l.groups, cpg]))
	return y.reshape(shape)
}

pub fn (mut l GroupNorm) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(l.channels), i32(l.groups)], [2])
	eps := mlx.f32_scalar(l.eps)
	_, vjps := mlx.vjp(gn_vjp_fn, [l.x, l.w, l.b, cfg, eps], [grad])
	l.dw = vjps[1]
	l.db = vjps[2]
	return vjps[0]
}

pub fn (mut l GroupNorm) params() []mlx.Array {
	return [l.w, l.b]
}

pub fn (mut l GroupNorm) grads() []mlx.Array {
	return [l.dw, l.db]
}

pub fn (mut l GroupNorm) set_params(ps []mlx.Array) {
	l.w = ps[0]
	l.b = ps[1]
}

pub fn (mut l GroupNorm) set_training(training bool) {}

pub fn (mut l GroupNorm) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w', l.w)
	m.insert('${prefix}.b', l.b)
}

pub fn (mut l GroupNorm) load_params(m mlx.MapStringToArray, prefix string) {
	l.w = reshape_to(m.get('${prefix}.w'), [l.channels], '${prefix}.w')
	l.b = reshape_to(m.get('${prefix}.b'), [l.channels], '${prefix}.b')
	l.w.eval()
	l.b.eval()
}
