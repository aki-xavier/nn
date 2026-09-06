// non-OOP — test file: free test_/helper functions are exempt from the OOP rule.
module nn

import mlx

// clifford_test.v — algebra tables, gradients and group-action geometry
// checks for the scalar/rotor/motor layers.

fn test_clifford_algebra_tables() {
	// rotor = quaternion algebra: i*j = k, j*i = -k, i² = j² = k² = -1
	r := Repr.rotor
	assert r.prod(r.basis(1), r.basis(2)) == [f32(0), 0, 0, 1]
	assert r.prod(r.basis(2), r.basis(1)) == [f32(0), 0, 0, -1]
	assert r.prod(r.basis(1), r.basis(1)) == [f32(-1), 0, 0, 0]
	assert r.prod([f32(1), 0, 0, 0], [f32(1), 0, 0, 0]) == [f32(1), 0, 0, 0]

	// motor = dual quaternion: e commutes, e² = 0
	m := Repr.motor
	assert m.prod(m.basis(4), m.basis(1)) == m.prod(m.basis(1), m.basis(4))
	assert m.prod(m.basis(4), m.basis(4)).all(it == 0)
	// (1 + e·i)(1 + e·j) = 1 + e(i + j)
	a := [f32(1), 0, 0, 0, 0, 1, 0, 0] // 1 + e·i
	b := [f32(1), 0, 0, 0, 0, 0, 1, 0] // 1 + e·j
	p := m.prod(a, b)
	assert p[0] == 1.0
	assert p[4] == 0.0 && p[5] == 1.0 && p[6] == 1.0

	// reverse flips vector parts only
	assert r.reverse([f32(1), 2, 3, 4]) == [f32(1), -2, -3, -4]
}

fn test_clifford_linear_gradient() {
	mut l := Layer(new_clifford_linear(.rotor, 2, 3, 77))
	x := mlx.array_f32([]f32{len: 2 * 2 * 4, init: f32((index * 5) % 7) / 6.0 - 0.5}, [
		2,
		2,
		4,
	])
	fd_check('cliff w', mut l, x, 0)
	fd_check('cliff b', mut l, x, 1)
}

fn test_group_layer_geometry() {
	// rotor: pi/2 about z rotates e1 to e2
	mut gl := Layer(new_group_layer(.rotor))
	r90 := mlx.array_f32([f32(0), 0, 1.5707963], [3]) // pi/2 about z
	gl.set_params([r90, mlx.zeros([3], .float32)])
	out := gl.forward(mlx.array_f32([f32(0), 1, 0, 0], [1, 4]))
	v := out.data_f32()
	assert absf(v[0]) < 1e-5 && absf(v[2]) > 0.99, 'rotor z90: e1 -> e2, got ${v}'

	// motor: identity rotation + translation t = (1,0,0) acting on the point
	// p = 1 + e·P (P = e1): textbook dual-quaternion identity is
	// p' = 1 + e·(R·P + t) = 1 + e·(2, 1, 0).
	mut gm := Layer(new_group_layer(.motor))
	gm.set_params([mlx.zeros([3], .float32), mlx.array_f32([f32(1), 0, 0], [3])])
	pt := mlx.array_f32([f32(1), 0, 0, 0, 0, 1, 0, 0], [1, 8])
	v2 := gm.forward(pt).data_f32()
	// P = x̂, t = (1,0,0) -> P + t = (2,0,0) at dual components 5..7
	assert absf(v2[0] - 1.0) < 1e-4, 'point scalar part should stay 1'
	assert absf(v2[4]) < 1e-4, 'dual scalar comp: ${v2}'
	assert absf(v2[5] - 2.0) < 1e-3, 'dual x should be 2: ${v2}'
	assert absf(v2[6]) < 1e-4, 'dual y should be 0: ${v2}'
	assert absf(v2[7]) < 1e-4

	// motor with rotation pi/2 about z, no translation: dual part = R·P = ŷ
	mut gm2 := Layer(new_group_layer(.motor))
	gm2.set_params([mlx.array_f32([f32(0), 0, 1.5707963], [3]), mlx.zeros([3], .float32)])
	v3 := gm2.forward(pt).data_f32()
	assert absf(v3[5]) < 1e-3 && absf(v3[6] - 1.0) < 1e-3, 'motor rotate: dual should be ŷ: ${v3}'
}

fn test_repr_switch() {
	mut rs := Layer(ReprSwitch{
		from: .scalar
		to: .rotor
	})
	x := mlx.array_f32([f32(1.5)], [1, 1])
	out := rs.forward(x)
	assert out.shape() == [1, 4]
	assert out.data_f32() == [f32(1.5), 0, 0, 0]

	mut rs2 := Layer(ReprSwitch{
		from: .rotor
		to: .motor
	})
	x2 := mlx.array_f32([f32(1), 2, 3, 4], [1, 4])
	out2 := rs2.forward(x2)
	assert out2.shape() == [1, 8]
	assert out2.data_f32()[0..4] == [f32(1), 2, 3, 4]
	assert out2.data_f32()[4..8] == [f32(0), 0, 0, 0]

	// backward through a paddle: 8-dim grad -> 4-dim keeps leading components
	gd := rs2.backward(mlx.array_f32([]f32{len: 8, init: f32(1 + index)}, [1, 8]))
	assert gd.shape() == [1, 4]
	assert gd.data_f32() == [f32(1), 2, 3, 4]
}
