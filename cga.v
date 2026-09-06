// non-OOP — cga_act_fwd must be a top-level fn: MLX autograd takes a plain
// C function pointer (mlx.Func) that cannot capture the layer instance.
module nn

import math
import mlx

// cga.v — the conformal group as a layer: Cl(4,1) unit rotors applied by
// conjugation x -> R·x·R̃.
//
// Parameters are the 10 bivector coordinates (basis bivectors in mask order
// [e12, e13, e14, e15, e23, e24, e25, e34, e35, e45]); the rotor is exp(B).
// Because the conformal group contains dilations (not norm-preserving), the
// dilation generator e4∧e5 (the last coordinate) is clipped to
// ±log(max_scale), keeping the layer bounded.

// cga_biv_masks lists the ten basis bivector masks for Cl(4,1).
const cga_biv_masks = [3, 5, 9, 17, 6, 10, 18, 12, 20, 24]

// bivector_from_params lays the 10 coordinates into the 32-dim slot layout
// (the mask list is not ascending, so scan the whole list per slot).
fn bivector_from_params(p mlx.Array) mlx.Array {
	mut parts := []mlx.Array{cap: 32}
	for i in 0 .. 32 {
		mut found := -1
		for j in 0 .. 10 {
			if cga_biv_masks[j] == i {
				found = j
				break
			}
		}
		if found >= 0 {
			parts << p.take_axis(mlx.array_i32([i32(found)], [1]), 0)
		} else {
			parts << mlx.zeros([1], .float32)
		}
	}
	return mlx.concatenate(parts, 0)
}

// reverse_cga flips the sign of grades 1/2/3 blades (index = bitmask).
pub fn reverse_cga(v mlx.Array) mlx.Array {
	mut sv := []f32{len: 32, init: 1.0}
	for i in 0 .. 32 {
		g := bit_count(i)
		if g == 1 || g == 2 || g == 3 {
			sv[i] = -1.0
		}
	}
	return v.multiply(mlx.array_f32(sv, [32]))
}

// cga_rotor builds exp(B) from the 10 bivector coordinates.  B² is a scalar
// s; the exponential splits into cosh/sinh (s >= 0) and cos/sin (s < 0)
// branches — both smooth at s = 0 (sinh(φ)/φ -> 1).
fn cga_rotor(params mlx.Array, max_scale f32) mlx.Array {
	mut p := params
	if max_scale > 0 {
		b := math.log(f64(max_scale))
		p = mlx.s_clip(params, -b, b)
	}
	b := bivector_from_params(p)
	t := table_array_cga()
	// s = <B B>_0 = T[0][k][j] B_k B_j (scalar part only: l = 0)
	s := mlx.einsum('lkj,k,j->l', [t, b, b]).take_axis(mlx.array_i32([i32(0)], [1]), 0).reshape([])
	phi := mlx.s_add(s.abs(), 1e-12).sqrt()
	k1 := mlx.where(mlx.s_ge(s, 0.0), phi.cosh(), phi.cos())
	k2 := mlx.where(mlx.s_ge(s, 0.0), phi.sinh().divide(phi), phi.sin().divide(phi))
	bk2 := b.multiply(k2)
	tail := bk2.take_axis(mlx.array_i32([]int{len: 31, init: index + 1}.map(i32(it)), [
		31,
	]), 0)
	return mlx.concatenate([k1.reshape([1]), tail], 0)
}

// cga_act applies R·x·R̃ with R built from params.
fn cga_act(x mlx.Array, params mlx.Array, max_scale f32) mlx.Array {
	r := cga_rotor(params, max_scale)
	t := table_array_cga()
	pv := mlx.einsum('...k, j, lkj -> ...l', [x, reverse_cga(r), t])
	return mlx.einsum('i, ...j, lij -> ...l', [r, pv, t])
}

// cga_act_fwd is the autograd trampoline; xs = [x, params, cfg] with cfg a
// 0-d float array carrying max_scale.
fn cga_act_fwd(xs []mlx.Array) []mlx.Array {
	return [cga_act(xs[0], xs[1], xs[2].item_f32())]
}

// CGAGroupLayer applies a conformal rotor (bounded dilation) by conjugation.
pub struct CGAGroupLayer {
pub:
	max_scale f32 = 8.0
mut:
	params  mlx.Array // [10] bivector coordinates
	x       mlx.Array
	dparams mlx.Array
}

pub fn new_cga_group_layer(max_scale f32) CGAGroupLayer {
	return CGAGroupLayer{
		max_scale: max_scale
		params: mlx.zeros([10], .float32)
	}
}

pub fn (mut l CGAGroupLayer) forward(x mlx.Array) mlx.Array {
	l.x = x
	return cga_act(x, l.params, l.max_scale)
}

pub fn (mut l CGAGroupLayer) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.f32_scalar(l.max_scale)
	_, vjps := mlx.vjp(cga_act_fwd, [l.x, l.params, cfg], [grad])
	l.dparams = vjps[1]
	return vjps[0]
}

pub fn (mut l CGAGroupLayer) params() []mlx.Array {
	return [l.params]
}

pub fn (mut l CGAGroupLayer) grads() []mlx.Array {
	return [l.dparams]
}

pub fn (mut l CGAGroupLayer) set_params(ps []mlx.Array) {
	l.params = ps[0]
}

pub fn (mut l CGAGroupLayer) set_training(training bool) {}

pub fn (mut l CGAGroupLayer) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.params', l.params)
}

pub fn (mut l CGAGroupLayer) load_params(m mlx.MapStringToArray, prefix string) {
	l.params = reshape_to(m.get('${prefix}.params'), [10], '${prefix}.params')
	l.params.eval()
}

// ============================================================================
// Geometry helpers: conformal embeddings and group-element builders.
// ============================================================================

// ConformalPoint embeds euclidean points as conformal points (p̂ = p + ½p²e∞ + e0).
pub fn conformal_point_pub(p mlx.Array) mlx.Array {
	return conformal_point(p)
}

// ExtractConformal normalises a (possibly scaled) conformal point back to
// euclidean coordinates; returns (points [n,3], scale λ).
pub fn extract_conformal_pub(x mlx.Array) (mlx.Array, mlx.Array) {
	return extract_point(x)
}

// cga_translation_params builds bivector coordinates for a translation
// rotor (B = -(a∧e∞)/2, sign fixed by the geometry tests).
pub fn cga_translation_params(a []f32, sign_k f32) mlx.Array {
	// a∧e∞ = a1(e14+e15) + a2(e24+e25) + a3(e34+e35): mask slots
	// 9, 17, 10, 18, 12, 20
	mut p := []f32{len: 10}
	p[2] = sign_k * a[0] // e14
	p[3] = sign_k * a[0] // e15
	p[4] = sign_k * a[1] // e23? no: mask order index 4 = 6 → e23 — but translation uses e24/e25:
	p[4] = 0
	p[5] = sign_k * a[1] // e24
	p[6] = sign_k * a[1] // e25
	p[7] = sign_k * a[2] // e34
	p[8] = sign_k * a[2] // e35
	return mlx.array_f32(p, [10])
}

// cga_rotation_params builds bivector coordinates for a rotation about a
// unit axis (B = (θ/2)·axis∧e? — see tests; rotation rotor uses e12-style).
pub fn cga_rotation_params(axis []f32, angle f32, sign_k f32) mlx.Array {
	mut p := []f32{len: 10}
	// rotation about x ~ e23 (mask 6, slot 4), about y ~ e13 (mask 5, slot 1),
	// about z ~ e12 (mask 3, slot 0)
	p[4] = sign_k * angle * 0.5 * axis[0]
	p[1] = sign_k * angle * 0.5 * axis[1]
	p[0] = sign_k * angle * 0.5 * axis[2]
	return mlx.array_f32(p, [10])
}

// cga_dilation_params builds bivector coordinates for a scale s (dilation
// rotor exp(B) with B along e4∧e5, mask 24).
pub fn cga_dilation_params(scale f32, sign_k f32) mlx.Array {
	mut p := []f32{len: 10}
	// D_s = exp(-(ln s)/2 · e45) reproduces p -> s·p (sign_k = -1)
	p[9] = sign_k * 0.5 * f32(math.log(f64(scale))) // e45
	return mlx.array_f32(p, [10])
}

// probe_rotor exposes the built rotor (debug aid).
pub fn probe_rotor(params mlx.Array, max_scale f32) mlx.Array {
	return cga_rotor(params, max_scale)
}

// probe_bivector exposes the assembled bivector (debug aid).
pub fn probe_bivector(params mlx.Array) mlx.Array {
	return bivector_from_params(params)
}
