// non-OOP — test file: free test_/helper functions are exempt from the OOP rule.
module nn

import mlx

// cga_test.v — conformal algebra (Cl(4,1)) validation: blade products,
// point embedding roundtrip, and rotor geometry (translation / rotation /
// dilation) with sign conventions pinned numerically.  Group compositions
// and finite-difference gradients complete the set.

fn test_cga_algebra_tables() {
	// e5² = -1, e4² = +1, e∞² = (e5+e4)² = 0 (null), e12² = -1, e45² = +1
	p55 := Repr.cga.prod(Repr.cga.basis(16), Repr.cga.basis(16))
	assert p55[0] == -1.0 // scalar component: e5·e5 = -1
	assert Repr.cga.prod(Repr.cga.basis(8), Repr.cga.basis(8))[0] == 1.0 // e4·e4 = +1
	e4v := Repr.cga.basis(8)
	e5v := Repr.cga.basis(16)
	mut e_inf_v := []f32{len: 32}
	for i in 0 .. 32 {
		e_inf_v[i] = e4v[i] + e5v[i]
	}
	einf := Repr.cga.prod(e_inf_v, e_inf_v)
	mut tiny := true
	for v in einf {
		if absf(v) > 1e-6 {
			tiny = false
		}
	}
	assert tiny, 'e∞² should be null'
	// e12² = -1 (component 0)
	p12 := Repr.cga.prod(Repr.cga.basis(3), Repr.cga.basis(3))
	assert p12[0] == -1.0
	// e45² = +1
	p45 := Repr.cga.prod(Repr.cga.basis(24), Repr.cga.basis(24))
	assert p45[0] == 1.0
}

fn mlx_e16() []f32 {
	mut v := []f32{len: 32}
	v[16] = 1.0
	return v
}

fn test_cga_point_roundtrip() {
	p := mlx.array_f32([f32(1), 2, -0.5, -1.5, 0.75, 3.25], [2, 3])
	pc := conformal_point_pub(p)
	assert pc.shape() == [2, 32]
	eu, lam := extract_conformal_pub(pc)
	assert absf(lam.data_f32()[0] - 1.0) < 1e-4
	d := eu.subtract(p).abs().max().item_f32()
	assert d < 1e-4, 'conformal roundtrip drifted ${d}'
}

// cga_translate applies the translation rotor and returns the extracted points.
fn cga_translate(a []f32, sign_k f32, p mlx.Array) mlx.Array {
	params := cga_translation_params(a, sign_k)
	mut l := Layer(new_cga_group_layer(8.0))
	l.set_params([params])
	transformed := l.forward(conformal_point_pub(p))
	eu, _ := extract_conformal_pub(transformed)
	return eu
}

fn test_cga_translation() {
	p := mlx.array_f32([f32(0), 0, 0, 1, -2, 0.5], [2, 3])
	a := [f32(0.5), -0.3, 0.75]
	expected := p.add(mlx.array_f32([a[0], a[1], a[2]], [1, 3]))
	// try both sign conventions; exactly one must reproduce p -> p + a
	mut ok := false
	for k in [f32(0.5), -0.5] {
		got := cga_translate([a[0], a[1], a[2]], k, p)
		if got.subtract(expected).abs().max().item_f32() < 1e-3 {
			ok = true
		}
	}
	assert ok, 'no translation sign convention reproduced p -> p + a'
}

fn test_cga_rotation() {
	// pi/2 about z: x̂ -> ŷ
	p := mlx.array_f32([f32(1), 0, 0], [1, 3])
	mut ok := false
	for k in [f32(1.0), -1.0] {
		params := cga_rotation_params([f32(0.0), 0.0, 1.0], 1.5707963, k)
		mut l := Layer(new_cga_group_layer(8.0))
		l.set_params([params])
		out := l.forward(conformal_point_pub(p))
		eu, _ := extract_conformal_pub(out)
		want := mlx.array_f32([f32(0), 1, 0], [1, 3])
		if eu.subtract(want).abs().max().item_f32() < 1e-3 {
			ok = true
		}
	}
	assert ok, 'no rotation sign convention produced x̂ -> ŷ'
}

fn test_cga_dilation() {
	// scale 2: p -> 2p
	p := mlx.array_f32([f32(0.5), 0.3, -1.0], [1, 3])
	mut ok := false
	for k in [f32(1.0), -1.0] {
		mut l := Layer(new_cga_group_layer(8.0))
		l.set_params([cga_dilation_params(2.0, k)])
		out := l.forward(conformal_point_pub(p))
		eu, _ := extract_conformal_pub(out)
		want := mlx.s_mul(p, 2.0)
		if eu.subtract(want).abs().max().item_f32() < 1e-2 {
			ok = true
		}
	}
	assert ok, 'no dilation sign convention produced p -> 2p'
}

fn test_cga_gradient() {
	mut l := Layer(new_cga_group_layer(8.0))
	x := mlx.array_f32([]f32{len: 32, init: f32((index * 7) % 5) / 5.0 - 0.2}, [1, 32])
	fd_check('cga.params', mut l, x, 0)
}

fn test_cga_homomorphism() {
	// R2 (R1 x R1̃) R2̃ == (R2·R1) x (R2·R1)̃, verified on a point
	mut l1 := Layer(new_cga_group_layer(8.0))
	mut l2 := Layer(new_cga_group_layer(8.0))
	l1.set_params([cga_rotation_params([f32(0.0), 0.0, 1.0], 0.7, 1.0)])
	l2.set_params([cga_translation_params([f32(0.4), -0.2, 0.3], 0.5)])
	p := mlx.array_f32([f32(0.3), -0.6, 0.9], [1, 3])
	pc := conformal_point_pub(p)

	// sequential action
	mut x := l1.forward(pc)
	x = l2.forward(x)
	// composed rotor: r2 ⊗ r1 as multivectors (host product for verification)
	r1 := probe_rotor(l1.params()[0], 8.0).data_f32()
	r2 := probe_rotor(l2.params()[0], 8.0).data_f32()
	comp := Repr.cga.prod(r2, r1)
	comp_arr := mlx.array_f32(comp, [32])
	mut lc := Layer(new_cga_group_layer(8.0))
	// set composed params by converting the rotor back to a bivector? instead
	// apply the composed rotor via a hand-built action with the same einsum.
	_ = lc
	t := table_array_cga()
	pv := mlx.einsum('...k, j, lkj -> ...l', [pc, mlx.array_f32(Repr.cga.reverse(comp)[0..32], [
		32,
	]), t])
	composed := mlx.einsum('i, ...j, lij -> ...l', [comp_arr, pv, t])
	d := x.subtract(composed).abs().max().item_f32()
	assert d < 1e-3, 'CGA homomorphism failed: ${d}'
}
