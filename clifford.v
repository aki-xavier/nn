// non-OOP — the vjp trampolines must be top-level fns: MLX autograd takes a
// plain C function pointer (mlx.Func) that cannot capture the layer instance.
module nn

import math
import mlx

// clifford.v — scalar / rotor / motor representation arena.
//
// One geometric product, three algebras, all selected by a `repr` field:
//   scalar: 1 dim (trivial algebra)
//   rotor:  4 dims (even subalgebra of Cl(0,3) = quaternions, group SO(3))
//   motor:  8 dims (even subalgebra of PGA Cl(3,0,1) = dual quaternions,
//           group SE(3); Chasles: every rigid motion is a screw)
//
// Each algebra is described by its structure-constants tensor T[l][k][j]:
// basis product e_k·e_j = sum_l T[l][k][j]·e_l.  CliffordLinear is a free
// (unconstrained) multivector-linear layer; GroupLayer applies a unit
// element by conjugation x -> g·x·g̃ (exponential-map parameterised), the
// versor layer with isometric Jacobians.  ReprSwitch lifts values between
// representations with embeddings that preserve the existing data.

// Repr selects the Clifford representation of a layer.
pub enum Repr {
	scalar
	rotor
	motor
}

// dim returns the number of basis components of the representation.
pub fn (r Repr) dim() int {
	return match r {
		.scalar { 1 }
		.rotor { 4 }
		.motor { 8 }
	}
}

// reverse returns the conjugation partner q* used in the sandwich
// x -> q·x·q*.  For rotors this is the quaternion conjugate; for motors it
// is the dual-quaternion conjugate r* - e·d* (signs on components
// [1,2,3,4]), which is the textbook SE(3) point-action convention.
pub fn (r Repr) reverse(v []f32) []f32 {
	mut out := v.clone()
	out[1] = -out[1]
	if r.dim() >= 3 {
		out[2] = -out[2]
	}
	if r.dim() >= 4 {
		out[3] = -out[3]
	}
	if r.dim() >= 8 {
		out[4] = -out[4]
	}
	return out
}

// table_array returns the structure constants as an mlx array [d, d, d]
// where dims are (l, k, j): e_k·e_j = sum_l T[l][k][j]·e_l.
pub fn (r Repr) table_array() mlx.Array {
	d := r.dim()
	mut t := []f32{len: d * d * d}
	for k in 0 .. d {
		for j in 0 .. d {
			p := r.prod(r.basis(k), r.basis(j))
			for l in 0 .. d {
				t[l * d * d + k * d + j] = p[l]
			}
		}
	}
	return mlx.array_f32(t, [d, d, d])
}

// basis returns basis element k as a unit vector.
fn (r Repr) basis(k int) []f32 {
	mut v := []f32{len: r.dim()}
	v[k] = 1.0
	return v
}

// prod is the geometric product on host flat vectors: quaternion product for
// rotors, dual quaternion product (e² = 0, e commutes) for motors.
pub fn (r Repr) prod(a []f32, b []f32) []f32 {
	match r {
		.scalar {
			return [a[0] * b[0]]
		}
		.rotor {
			return quat_prod(a, b)
		}
		.motor {
			a_r := a[0..4]
			a_d := a[4..8]
			b_r := b[0..4]
			b_d := b[4..8]
			// dual part: a_d⊗b_r + a_r⊗b_d (element-wise), then append to rotation part
			mut dual := []f32{len: 4}
			dr := quat_prod(a_d, b_r)
			rd := quat_prod(a_r, b_d)
			for i in 0 .. 4 {
				dual[i] = dr[i] + rd[i]
			}
			mut out := quat_prod(a_r, b_r)
			out << dual
			return out
		}
	}
}

// quat_prod multiplies Hamilton quaternions [w, x, y, z].
fn quat_prod(a []f32, b []f32) []f32 {
	return [
		a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
		a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
		a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
		a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
	]
}

// ============================================================================
// CliffordLinear: free multivector weights, geometric-product layer.
// x: [..., in_dim, R], w: [out_dim, in_dim, R], bias: [out_dim, R]
// ============================================================================

pub struct CliffordLinear {
pub:
	repr    Repr
	in_dim  int
	out_dim int
mut:
	w  mlx.Array // [out_dim, in_dim, R]
	b  mlx.Array // [out_dim, R]
	x  mlx.Array // input cached by forward
	dw mlx.Array
	db mlx.Array
}

pub fn new_clifford_linear(repr Repr, in_dim int, out_dim int, seed u64) CliffordLinear {
	r := repr.dim()
	key := mlx.random_key(seed)
	defer {
		key.free()
	}
	scale := f32(1.0 / math.sqrt(f64(in_dim * r)))
	return CliffordLinear{
		repr: repr
		in_dim: in_dim
		out_dim: out_dim
		w: mlx.random_normal([out_dim, in_dim, r], .float32, 0.0, scale, key)
		b: mlx.zeros([out_dim, r], .float32)
	}
}

// clifford_linear_fwd is the autograd trampoline; xs carries [x, w, b, cfg]
// with cfg an int32 array [repr_id, in_dim, out_dim].
fn clifford_linear_fwd(xs []mlx.Array) []mlx.Array {
	cfg := xs[3].data_i32()
	repr := repr_of(cfg[0])
	t := repr.table_array()
	// y[...,o,l] = sum_{i,j,k} x[...,i,j] w[o,i,k] T[l,k,j]
	return [mlx.einsum('...ij, oik, lkj -> ...ol', [xs[0], xs[1], t]).add(xs[2])]
}

pub fn (mut l CliffordLinear) forward(x mlx.Array) mlx.Array {
	l.x = x
	t := l.repr.table_array()
	y := mlx.einsum('...ij, oik, lkj -> ...ol', [x, l.w, t])
	return y.add(l.b)
}

pub fn (mut l CliffordLinear) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(int(l.repr)), i32(l.in_dim), i32(l.out_dim)], [3])
	_, vjps := mlx.vjp(clifford_linear_fwd, [l.x, l.w, l.b, cfg], [grad])
	l.dw = vjps[1]
	l.db = vjps[2]
	return vjps[0]
}

pub fn (mut l CliffordLinear) params() []mlx.Array {
	return [l.w, l.b]
}

pub fn (mut l CliffordLinear) grads() []mlx.Array {
	return [l.dw, l.db]
}

pub fn (mut l CliffordLinear) set_params(ps []mlx.Array) {
	l.w = ps[0]
	l.b = ps[1]
}

pub fn (mut l CliffordLinear) set_training(training bool) {}

pub fn (mut l CliffordLinear) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w', l.w)
	m.insert('${prefix}.b', l.b)
}

pub fn (mut l CliffordLinear) load_params(m mlx.MapStringToArray, prefix string) {
	l.w = reshape_to(m.get('${prefix}.w'), [l.out_dim, l.in_dim, l.repr.dim()], '${prefix}.w')
	l.b = reshape_to(m.get('${prefix}.b'), [l.out_dim, l.repr.dim()], '${prefix}.b')
	l.w.eval()
	l.b.eval()
}

// repr_of maps the config id back to a Repr.
pub fn repr_of(id int) Repr {
	return unsafe { Repr(id) }
}

// ============================================================================
// GroupLayer: unit element (rotor or motor) applied by conjugation.
// Params live in Lie-algebra coordinates (rotvec 3, transvec 3 for motors)
// and are mapped through the exponential map, so the element stays unit and
// the conjugation Jacobian stays isometric.
// ============================================================================

pub struct GroupLayer {
pub:
	repr Repr
mut:
	rotvec mlx.Array // [3]
	trans  mlx.Array // [3]
	x      mlx.Array // input cached by forward
	drot   mlx.Array
	dtrans mlx.Array
}

pub fn new_group_layer(repr Repr) GroupLayer {
	return GroupLayer{
		repr: repr
		rotvec: mlx.zeros([3], .float32)
		trans: mlx.zeros([3], .float32)
	}
}

// build_element computes the unit element from Lie-algebra coordinates,
// using array ops throughout so the exp map is differentiable.  The rotation
// norm is smoothed as theta = sqrt(|x|² + eps²) so gradients stay bounded at
// the origin (|x| itself is not differentiable there).
fn build_element(repr Repr, rotvec mlx.Array, trans mlx.Array) mlx.Array {
	theta := mlx.s_add(rotvec.square().sum(), 1e-12).sqrt()
	h := theta.multiply(mlx.f32_scalar(0.5))
	s := h.sin()
	c := h.cos()
	// n = rotvec / theta (bounded by 1 thanks to the smoothed norm)
	n := rotvec.divide(theta)
	r := mlx.concatenate([c.reshape([1]), n.multiply(s.reshape([1])).reshape([3])], 0)
	if repr == .scalar {
		return c.reshape([1])
	}
	if repr == .rotor {
		return r
	}
	// dual quaternion: g = r + e/2·(t⊗r_), using the quaternion table for the
	// pure-quaternion product t·r (t = [0, tx, ty, tz]).
	tq := mlx.concatenate([mlx.zeros([1], .float32), trans.reshape([3])], 0)
	tqr := Repr.rotor.table_array()
	d := mlx.einsum('i, j, lij -> l', [tq, r, tqr]).multiply(mlx.f32_scalar(0.5))
	return mlx.concatenate([r, d], 0)
}

// reverse_arr flips the vector-part signs of a multivector array.
fn reverse_arr(repr Repr, g mlx.Array) mlx.Array {
	mut sv := []f32{len: repr.dim(), init: 1.0}
	for i in 0 .. repr.dim() {
		mut v := []f32{len: repr.dim()}
		v[i] = 1.0
		if repr.reverse(v)[i] < 0 {
			sv[i] = -1.0
		}
	}
	return g.multiply(mlx.array_f32(sv, [repr.dim()]))
}

// group_act_fwd is the autograd trampoline; xs carries [x, rotvec, trans,
// cfg] with cfg an int32 array [repr_id, _].
fn group_act_fwd(xs []mlx.Array) []mlx.Array {
	repr := repr_of(xs[3].data_i32()[0])
	return [group_act(xs[0], xs[1], xs[2], repr)]
}

// group_act applies g·x·g̃ (g built from rotvec/trans via the exp map).
fn group_act(x mlx.Array, rotvec mlx.Array, trans mlx.Array, repr Repr) mlx.Array {
	g := build_element(repr, rotvec, trans)
	t := repr.table_array()
	p := mlx.einsum('...k, j, lkj -> ...l', [x, reverse_arr(repr, g), t])
	return mlx.einsum('i, ...j, lij -> ...l', [g, p, t])
}

pub fn (mut l GroupLayer) forward(x mlx.Array) mlx.Array {
	l.x = x
	return group_act(x, l.rotvec, l.trans, l.repr)
}

pub fn (mut l GroupLayer) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(int(l.repr)), i32(0)], [2])
	_, vjps := mlx.vjp(group_act_fwd, [l.x, l.rotvec, l.trans, cfg], [grad])
	l.drot = vjps[1]
	l.dtrans = vjps[2]
	return vjps[0]
}

pub fn (mut l GroupLayer) params() []mlx.Array {
	return [l.rotvec, l.trans]
}

pub fn (mut l GroupLayer) grads() []mlx.Array {
	return [l.drot, l.dtrans]
}

pub fn (mut l GroupLayer) set_params(ps []mlx.Array) {
	l.rotvec = ps[0]
	l.trans = ps[1]
}

pub fn (mut l GroupLayer) set_training(training bool) {}

pub fn (mut l GroupLayer) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.rotvec', l.rotvec)
	m.insert('${prefix}.trans', l.trans)
}

pub fn (mut l GroupLayer) load_params(m mlx.MapStringToArray, prefix string) {
	l.rotvec = reshape_to(m.get('${prefix}.rotvec'), [3], '${prefix}.rotvec')
	l.trans = reshape_to(m.get('${prefix}.trans'), [3], '${prefix}.trans')
	l.rotvec.eval()
	l.trans.eval()
}

// ============================================================================
// ReprSwitch: lift values across representations, preserving the existing
// components (scalar -> rotor -> motor are injective embeddings).  The
// inverse direction keeps the leading components.
// ============================================================================

pub struct ReprSwitch {
pub:
	from Repr
	to   Repr
mut:
	shape     []int
	from_kept int // cached for the backward pass
}

pub fn (mut l ReprSwitch) forward(x mlx.Array) mlx.Array {
	l.shape = x.shape()
	if l.from == l.to {
		return x
	}
	last := x.shape().len - 1
	f := l.from.dim()
	t := l.to.dim()
	if t >= f {
		l.from_kept = f
		mut zshape := x.shape()
		zshape[last] = t - f
		z := mlx.zeros(zshape, .float32)
		return mlx.concatenate([x, z], -1)
	}
	// downgrade: keep the leading `t` components
	idx := mlx.array_i32([]int{len: t, init: index}.map(i32(it)), [t])
	l.from_kept = t
	return x.take_axis(idx, last)
}

pub fn (mut l ReprSwitch) backward(grad mlx.Array) mlx.Array {
	last := grad.shape().len - 1
	f := l.from.dim()
	t := l.to.dim()
	if t >= f {
		idx := mlx.array_i32([]int{len: f, init: index}.map(i32(it)), [f])
		return grad.take_axis(idx, last)
	}
	// downgrade forward: pad the kept `t` components with zeros back to f
	mut zshape := grad.shape()
	zshape[last] = f - t
	z := mlx.zeros(zshape, .float32)
	return mlx.concatenate([grad.take_axis(mlx.array_i32([]int{len: t, init: index}.map(i32(it)), [
		t,
	]), last), z], -1)
}

pub fn (mut l ReprSwitch) params() []mlx.Array {
	return []
}

pub fn (mut l ReprSwitch) grads() []mlx.Array {
	return []
}

pub fn (mut l ReprSwitch) set_params(ps []mlx.Array) {}

pub fn (mut l ReprSwitch) set_training(training bool) {}

pub fn (mut l ReprSwitch) save_params(m mlx.MapStringToArray, prefix string) {}

pub fn (mut l ReprSwitch) load_params(m mlx.MapStringToArray, prefix string) {}
