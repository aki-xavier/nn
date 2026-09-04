// non-OOP — the vjp trampolines must be top-level fns: MLX autograd takes a
// plain C function pointer (mlx.Func) that cannot capture the layer instance.
module nn

import mlx

// norm.v — normalisation layers.  Backward passes run through MLX autograd
// (vjp): normalisation gradients involve coupled mean/variance terms that
// are error-prone to hand-write.

// LayerNorm normalises the last axis with per-channel affine parameters.
// Input is NHWC ([..., c]); weight/bias have shape [c].
pub struct LayerNorm {
pub:
	channels int
	eps      f32 = 1e-5
mut:
	w  mlx.Array // gamma [c]
	b  mlx.Array // beta [c]
	x  mlx.Array // input cached by forward
	dw mlx.Array
	db mlx.Array
}

pub fn new_layer_norm(channels int) LayerNorm {
	return LayerNorm{
		channels: channels
		w: mlx.ones([channels], .float32)
		b: mlx.zeros([channels], .float32)
	}
}

// ln_vjp_fn is the autograd trampoline for LayerNorm.backward; xs carries
// [x, gamma, beta, eps] with eps a 0-d float array.
fn ln_vjp_fn(xs []mlx.Array) []mlx.Array {
	eps := xs[3].item_f32()
	return [xs[0].layer_norm(xs[1], xs[2], eps)]
}

pub fn (mut l LayerNorm) forward(x mlx.Array) mlx.Array {
	l.x = x
	return x.layer_norm(l.w, l.b, l.eps)
}

pub fn (mut l LayerNorm) backward(grad mlx.Array) mlx.Array {
	eps := mlx.f32_scalar(l.eps)
	_, vjps := mlx.vjp(ln_vjp_fn, [l.x, l.w, l.b, eps], [grad])
	l.dw = vjps[1]
	l.db = vjps[2]
	return vjps[0]
}

pub fn (mut l LayerNorm) params() []mlx.Array {
	return [l.w, l.b]
}

pub fn (mut l LayerNorm) grads() []mlx.Array {
	return [l.dw, l.db]
}

pub fn (mut l LayerNorm) set_params(ps []mlx.Array) {
	l.w = ps[0]
	l.b = ps[1]
}

pub fn (mut l LayerNorm) set_training(training bool) {}

pub fn (mut l LayerNorm) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w', l.w)
	m.insert('${prefix}.b', l.b)
}

pub fn (mut l LayerNorm) load_params(m mlx.MapStringToArray, prefix string) {
	l.w = m.get('${prefix}.w')
	l.b = m.get('${prefix}.b')
	l.w.eval()
	l.b.eval()
}

// BatchNorm2d normalises NHWC inputs per channel over the [n, h, w] axes.
// Training uses batch statistics and updates running estimates; inference
// uses the running estimates.
pub struct BatchNorm2d {
pub:
	channels int
	eps      f32 = 1e-5
	momentum f32 = 0.1
mut:
	w           mlx.Array // gamma [1, 1, 1, c]
	b           mlx.Array // beta  [1, 1, 1, c]
	x           mlx.Array // input cached by forward
	dw          mlx.Array
	db          mlx.Array
	running_mu  mlx.Array // [1, 1, 1, c]
	running_var mlx.Array // [1, 1, 1, c]
	training    bool = true
}

pub fn new_batch_norm2d(channels int) BatchNorm2d {
	shape := [1, 1, 1, channels]
	return BatchNorm2d{
		channels: channels
		w: mlx.ones(shape, .float32)
		b: mlx.zeros(shape, .float32)
		running_mu: mlx.zeros(shape, .float32)
		running_var: mlx.ones(shape, .float32)
	}
}

// bn_vjp_fn is the autograd trampoline for BatchNorm2d.backward; xs carries
// [x, gamma, beta, eps] and recomputes the batch statistics.
fn bn_vjp_fn(xs []mlx.Array) []mlx.Array {
	eps := xs[3].item_f32()
	mu := xs[0].mean_axes([0, 1, 2], true)
	xc := xs[0].subtract(mu)
	var_ := xc.square().mean_axes([0, 1, 2], true)
	return [
		xc.divide(var_.add(mlx.f32_scalar(eps)).sqrt()).multiply(xs[1]).add(xs[2]),
	]
}

pub fn (mut l BatchNorm2d) forward(x mlx.Array) mlx.Array {
	l.x = x
	if l.training {
		mu := x.mean_axes([0, 1, 2], true)
		xc := x.subtract(mu)
		var_ := xc.square().mean_axes([0, 1, 2], true)
		// update running estimates with detached (stop_gradient) statistics
		l.running_mu = mlx.s_mul(l.running_mu, f64(1.0 - l.momentum)).add(mlx.s_mul(mu.stop_gradient(), f64(l.momentum)))
		l.running_var = mlx.s_mul(l.running_var, f64(1.0 - l.momentum)).add(mlx.s_mul(var_.stop_gradient(), f64(l.momentum)))
		return xc.divide(var_.add(mlx.f32_scalar(l.eps)).sqrt()).multiply(l.w).add(l.b)
	}
	return x.subtract(l.running_mu).divide(l.running_var.add(mlx.f32_scalar(l.eps)).sqrt()).multiply(l.w).add(l.b)
}

pub fn (mut l BatchNorm2d) backward(grad mlx.Array) mlx.Array {
	eps := mlx.f32_scalar(l.eps)
	_, vjps := mlx.vjp(bn_vjp_fn, [l.x, l.w, l.b, eps], [grad])
	l.dw = vjps[1]
	l.db = vjps[2]
	return vjps[0]
}

pub fn (mut l BatchNorm2d) params() []mlx.Array {
	return [l.w, l.b]
}

pub fn (mut l BatchNorm2d) grads() []mlx.Array {
	return [l.dw, l.db]
}

pub fn (mut l BatchNorm2d) set_params(ps []mlx.Array) {
	l.w = ps[0]
	l.b = ps[1]
}

pub fn (mut l BatchNorm2d) set_training(training bool) {
	l.training = training
}

pub fn (mut l BatchNorm2d) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w', l.w)
	m.insert('${prefix}.b', l.b)
	m.insert('${prefix}.running_mu', l.running_mu)
	m.insert('${prefix}.running_var', l.running_var)
}

pub fn (mut l BatchNorm2d) load_params(m mlx.MapStringToArray, prefix string) {
	l.w = m.get('${prefix}.w')
	l.b = m.get('${prefix}.b')
	l.running_mu = m.get('${prefix}.running_mu')
	l.running_var = m.get('${prefix}.running_var')
	l.w.eval()
	l.b.eval()
	l.running_mu.eval()
	l.running_var.eval()
}
