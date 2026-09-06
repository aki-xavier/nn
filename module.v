module nn

import mlx

// module.v — Module: the generic compositional container.
//
// While Sequential is the linear chain, Module nests any Layer (including
// other Modules) and recurses every protocol method (forward/backward/
// params/grads/set_params/set_training/save/load), so custom blocks can be
// composed without touching the Layer sum type.  named_parameters() exposes
// the flat parameter list with dotted names ('child.3.w') for checkpoint
// mapping.

pub struct Module {
mut:
	children []Layer
}

// add appends a child layer or module.
pub fn (mut m Module) add(l Layer) {
	m.children << l
}

// children_len returns the number of children.
pub fn (m Module) children_len() int {
	return m.children.len
}

pub fn (mut m Module) forward(x mlx.Array) mlx.Array {
	mut out := x
	for mut c in m.children {
		out = c.forward(out)
	}
	return out
}

pub fn (mut m Module) backward(grad mlx.Array) mlx.Array {
	mut g := grad
	for i := m.children.len - 1; i >= 0; i-- {
		g = m.children[i].backward(g)
	}
	return g
}

pub fn (mut m Module) set_training(training bool) {
	for mut c in m.children {
		c.set_training(training)
	}
}

pub fn (mut m Module) params() []mlx.Array {
	mut out := []mlx.Array{}
	for mut c in m.children {
		out << c.params()
	}
	return out
}

pub fn (mut m Module) grads() []mlx.Array {
	mut out := []mlx.Array{}
	for mut c in m.children {
		out << c.grads()
	}
	return out
}

pub fn (mut m Module) set_params(ps []mlx.Array) {
	mut off := 0
	for mut c in m.children {
		n := c.params().len
		if n > 0 {
			c.set_params(ps[off..off + n])
		}
		off += n
	}
}

// named_parameters returns a dotted-name -> parameter map (child indices
// separate path segments, e.g. 'child.1.w').
pub fn (mut m Module) named_parameters() map[string]mlx.Array {
	mut out := map[string]mlx.Array{}
	fill_named(mut m.children, 'child', mut out)
	return out
}

fn fill_named(mut children []Layer, prefix string, mut out map[string]mlx.Array) {
	for i, mut c in children {
		key := '${prefix}.${i}'
		match mut c {
			Module {
				for k, v in c.named_parameters() {
					out['${key}.${k}'] = v
				}
			}
			Linear, Conv2d, Conv1d, Conv3d, LayerNorm, GroupNorm, CliffordLinear {
				ps := c.params()
				for j, p in ps {
					out['${key}.${['w', 'b'][j]}'] = p
				}
			}
			BatchNorm2d {
				ps := c.params()
				out['${key}.w'] = ps[0]
				out['${key}.b'] = ps[1]
				out['${key}.running_mu'] = c.running_mu
				out['${key}.running_var'] = c.running_var
			}
			MotorGroupLayer {
				ps := c.params()
				out['${key}.q'] = ps[0]
				out['${key}.t'] = ps[1]
			}
			GroupLayer {
				ps := c.params()
				out['${key}.rotvec'] = ps[0]
				out['${key}.trans'] = ps[1]
			}
			else {}
		}
	}
}

pub fn (mut mod_ Module) save_params(m mlx.MapStringToArray, prefix string) {
	for i, mut c in mod_.children {
		c.save_params(m, '${prefix}.child.${i}')
	}
}

pub fn (mut mod_ Module) load_params(m mlx.MapStringToArray, prefix string) {
	for i, mut c in mod_.children {
		c.load_params(m, '${prefix}.child.${i}')
	}
}
