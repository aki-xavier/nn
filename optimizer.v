module nn

import math
import mlx

// optimizer.v — parameter update rules.  Optimizers work generically through
// the Layer params()/grads()/set_params() protocol, so any optimizer can be
// used with any layer.

pub type Optimizer = Adam | SGD

// step applies one update to every layer (sum-type dispatch).
pub fn (mut o Optimizer) step(mut layers []Layer) {
	match mut o {
		SGD { o.step(mut layers) }
		Adam { o.step(mut layers) }
	}
}

// SGD is plain stochastic gradient descent with a fixed learning rate.
pub struct SGD {
pub:
	lr f32
}

// step applies one gradient-descent update to every layer.
pub fn (o SGD) step(mut layers []Layer) {
	for mut l in layers {
		ps := l.params()
		gs := l.grads()
		if ps.len == 0 {
			continue
		}
		mut next := []mlx.Array{cap: ps.len}
		for j in 0 .. ps.len {
			next << ps[j].subtract(mlx.s_mul(gs[j], f64(o.lr)))
		}
		l.set_params(next)
	}
}

// Adam is the Adam optimizer with bias-corrected first/second moments.
pub struct Adam {
pub:
	lr    f32 = 1e-3
	beta1 f32 = 0.9
	beta2 f32 = 0.999
	eps   f32 = 1e-8
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
			g := gs[j]
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
