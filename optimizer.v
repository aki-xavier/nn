module nn

import math
import mlx

// optimizer.v — parameter update rules.  Optimizers work generically through
// the Layer params()/grads()/set_params() protocol, so any optimizer can be
// used with any layer.  Both optimizers support global-norm gradient
// clipping (`clip_norm`, 0 = disabled).

pub type Optimizer = Adam | SGD

// step applies one update to every layer (sum-type dispatch).
pub fn (mut o Optimizer) step(mut layers []Layer) {
	match mut o {
		SGD { o.step(mut layers) }
		Adam { o.step(mut layers) }
	}
}

// set_lr replaces the learning rate (used by schedulers).
pub fn (mut o Optimizer) set_lr(lr f32) {
	match mut o {
		SGD {
			o.lr = lr
		}
		Adam {
			o.lr = lr
		}
	}
}

// save_state persists optimizer state (Adam moments); SGD is stateless.
pub fn (mut o Optimizer) save_state(path string) {
	match mut o {
		Adam { o.save_state(path) }
		SGD {}
	}
}

// load_state restores optimizer state (Adam moments) from a checkpoint.
pub fn (mut o Optimizer) load_state(path string) {
	match mut o {
		Adam { o.load_state(path) }
		SGD {}
	}
}

// SGD is plain stochastic gradient descent with a fixed learning rate.
// `clip_norm > 0` applies global-norm gradient clipping.
pub struct SGD {
pub:
	clip_norm f32 = 0
pub mut:
	lr f32
}

// step applies one gradient-descent update to every layer.
pub fn (o SGD) step(mut layers []Layer) {
	norm := global_norm(mut layers)
	scale := clip_factor(norm, o.clip_norm)
	for mut l in layers {
		ps := l.params()
		gs := l.grads()
		if ps.len == 0 {
			continue
		}
		mut next := []mlx.Array{cap: ps.len}
		for j in 0 .. ps.len {
			next << ps[j].subtract(mlx.s_mul(mlx.s_mul(gs[j], f64(scale)), f64(o.lr)))
		}
		l.set_params(next)
	}
}

// Adam is the Adam optimizer with bias-corrected first/second moments.
// `clip_norm > 0` applies global-norm gradient clipping.
pub struct Adam {
pub:
	beta1     f32 = 0.9
	beta2     f32 = 0.999
	eps       f32 = 1e-8
	clip_norm f32 = 0
pub mut:
	lr f32 = 1e-3
mut:
	t int
	m []mlx.Array
	v []mlx.Array
}

// step applies one Adam update to every layer.
pub fn (mut o Adam) step(mut layers []Layer) {
	o.t++
	bc1 := 1.0 - math.pow(f64(o.beta1), f64(o.t))
	bc2 := 1.0 - math.pow(f64(o.beta2), f64(o.t))
	norm := global_norm(mut layers)
	scale := clip_factor(norm, o.clip_norm)
	mut slot := 0
	for mut l in layers {
		ps := l.params()
		gs := l.grads()
		if ps.len == 0 {
			continue
		}
		for o.m.len < slot + ps.len {
			o.m << mlx.zeros_like(ps[o.m.len - slot])
			o.v << mlx.zeros_like(ps[o.v.len - slot])
		}
		mut next := []mlx.Array{cap: ps.len}
		for j in 0 .. ps.len {
			g := mlx.s_mul(gs[j], f64(scale))
			mut mj := o.m[slot + j]
			mut vj := o.v[slot + j]
			mj = mlx.s_mul(mj, f64(o.beta1)).add(mlx.s_mul(g, 1.0 - f64(o.beta1)))
			vj = mlx.s_mul(vj, f64(o.beta2)).add(mlx.s_mul(g.square(), 1.0 - f64(o.beta2)))
			o.m[slot + j] = mj
			o.v[slot + j] = vj
			mhat := mlx.s_mul(mj, 1.0 / bc1)
			vhat := mlx.s_mul(vj, 1.0 / bc2)
			p := ps[j].subtract(mlx.s_mul(mhat.divide(vhat.sqrt().add(mlx.f32_scalar(o.eps))), f64(o.lr)))
			next << p
		}
		l.set_params(next)
		slot += ps.len
	}
}

// save_state persists the Adam moment state (t, m, v) so training can resume.
pub fn (mut o Adam) save_state(path string) {
	m := mlx.new_map_string_to_array()
	m.insert('t', mlx.array_i32([i32(o.t)], [1]))
	for i in 0 .. o.m.len {
		m.insert('m.${i}', o.m[i])
		m.insert('v.${i}', o.v[i])
	}
	meta := mlx.new_map_string_to_string()
	mlx.save_safetensors(path, m, meta)
	m.free()
	meta.free()
}

// load_state restores the Adam moment state (arrays are CPU-materialised).
pub fn (mut o Adam) load_state(path string) {
	shapes := parse_safetensors_header(path)
	m, meta := mlx.load_safetensors(path)
	o.t = m.get('t').item_i32()
	o.m = []
	o.v = []
	for name in shapes.keys() {
		if name.starts_with('m.') {
			idx := name.all_after('m.')
			o.m << m.get(name)
			o.v << m.get('v.${idx}')
		}
	}
	m.free()
	meta.free()
}

// ============================================================================
// Learning-rate schedulers.
// ============================================================================

pub type LRScheduler = CosineLR | StepLR

// rate returns the learning rate for epoch `epoch` (0-based).
pub fn (mut s LRScheduler) rate(epoch int) f32 {
	match mut s {
		StepLR {
			return s.rate(epoch)
		}
		CosineLR {
			return s.rate(epoch)
		}
	}
}

// set_scheduler attaches a scheduler to an optimizer (sets lr for epoch 0).
pub fn (mut o Optimizer) set_scheduler(mut s LRScheduler) {
	o.set_lr(s.rate(0))
}

// advance updates the optimizer lr to the given epoch.
pub fn (mut o Optimizer) schedule(epoch int, mut s LRScheduler) {
	o.set_lr(s.rate(epoch))
}

// StepLR multiplies the lr by `gamma` every `step_size` epochs.
pub struct StepLR {
pub:
	lr        f32
	step_size int
	gamma     f32 = 0.1
}

pub fn (mut s StepLR) rate(epoch int) f32 {
	decays := epoch / s.step_size
	return s.lr * f32(math.pow(f64(s.gamma), f64(decays)))
}

// CosineLR anneals from `lr_max` to `lr_min` with a cosine curve.
pub struct CosineLR {
pub:
	lr_max f32
	lr_min f32 = 0
	t_max  int
}

pub fn (mut s CosineLR) rate(epoch int) f32 {
	mut t := epoch % s.t_max
	return s.lr_min + 0.5 * (s.lr_max - s.lr_min) * (1.0 + f32(math.cos(f64(t) * math.pi / f64(s.t_max))))
}
