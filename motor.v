module nn

import mlx

// motor.v — MotorGroupLayer: the group layer hard-wired to SE(3), the
// production form for rigid-motion tasks.
//
// The rotation is parameterised by a *raw* quaternion that is normalised
// inside the forward pass — smooth everywhere, so there is no exponential-map
// singularity at the origin.  The backward pass is analytic (no vjp, no
// structure-constants table): on point-form inputs 1 + e·(0, P, 0) the layer
// computes 1 + e·(0, R·P + t, 0), and the closed-form derivatives of
// R (from the unit quaternion) give the parameter gradients directly.
//
// Inputs/outputs are motor-repr tensors [n, 8]; only the dual vector
// components (5..7) carry the point, and they are also the only link the
// gradient flows through.

pub struct MotorGroupLayer {
mut:
	q  mlx.Array // raw quaternion [4]
	t  mlx.Array // translation [3]
	x  mlx.Array // input cached by forward
	dq mlx.Array // [4]
	dt mlx.Array // [3]
}

pub fn new_motor_group_layer() MotorGroupLayer {
	return MotorGroupLayer{
		q: mlx.array_f32([f32(1), 0, 0, 0], [4])
		t: mlx.zeros([3], .float32)
	}
}

// cross3 computes the 3-d cross product a × b for [n, 3] arrays (a may be
// [1, 3], broadcast by MLX).
fn cross3(a mlx.Array, b mlx.Array) mlx.Array {
	a0 := a.take_axis(mlx.array_i32([i32(0)], [1]), 1)
	a1 := a.take_axis(mlx.array_i32([i32(1)], [1]), 1)
	a2 := a.take_axis(mlx.array_i32([i32(2)], [1]), 1)
	b0 := b.take_axis(mlx.array_i32([i32(0)], [1]), 1)
	b1 := b.take_axis(mlx.array_i32([i32(1)], [1]), 1)
	b2 := b.take_axis(mlx.array_i32([i32(2)], [1]), 1)
	c0 := a1.multiply(b2).subtract(a2.multiply(b1))
	c1 := a2.multiply(b0).subtract(a0.multiply(b2))
	c2 := a0.multiply(b1).subtract(a1.multiply(b0))
	return mlx.concatenate([c0, c1, c2], 1)
}

// eps_tensor returns ε_{ijk} as a flat [27] array.
fn eps_tensor() []f32 {
	mut e := []f32{len: 27}
	for i in 0 .. 3 {
		for j in 0 .. 3 {
			for k in 0 .. 3 {
				mut val := 0
				if (i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) || (i == 2 && j == 0 && k == 1) {
					val = 1
				}
				if (i == 0 && j == 2 && k == 1) || (i == 2 && j == 1 && k == 0) || (i == 1 && j == 0 && k == 2) {
					val = -1
				}
				e[i * 9 + j * 3 + k] = f32(val)
			}
		}
	}
	return e
}

// rot_matrix builds the rotation matrix R of a unit (or raw) quaternion.
fn rot_matrix(q mlx.Array) mlx.Array {
	s := mlx.s_add(q.square().sum(), 1e-12).sqrt()
	qn := q.divide(s)
	w := qn.take_axis(mlx.array_i32([i32(0)], [1]), 0)
	v := qn.take_axis(mlx.array_i32([i32(1), 2, 3], [3]), 0)
	// einsum 'ijk,k->ij' gives K = -[v]x; negate so K u = v x u and the
	// standard quaternion rotation matrix holds.
	k := mlx.s_mul(mlx.einsum('ijk,k->ij', [mlx.array_f32(eps_tensor(), [3, 3, 3]), v]), -1.0)
	k2 := mlx.einsum('ij,jk->ik', [k, k])
	i3 := mlx.array_f32(identity3(), [3, 3])
	return i3.add(k2.multiply(mlx.f32_scalar(2.0))).add(k.multiply(w.multiply(mlx.f32_scalar(2.0))))
}

fn identity3() []f32 {
	mut m := []f32{len: 9}
	for i in 0 .. 3 {
		m[i * 3 + i] = 1.0
	}
	return m
}

fn identity34() []f32 {
	mut m := []f32{len: 12}
	for i in 0 .. 3 {
		m[i * 4 + i + 1] = 1.0
	}
	return m
}

// flatten_leading reshapes x [..., 8] into [M, 8] where M = prod(leading).
fn flatten_leading(x mlx.Array) mlx.Array {
	shape := x.shape()
	mut m := 1
	for d in shape[..shape.len - 1] {
		m *= d
	}
	return x.reshape([m, 8])
}

pub fn (mut l MotorGroupLayer) forward(x mlx.Array) mlx.Array {
	l.x = x
	xm := flatten_leading(x)
	n := xm.dim(0)
	p := xm.take_axis(mlx.array_i32([i32(5), 6, 7], [3]), 1)
	rp := mlx.einsum('ij,nj->ni', [rot_matrix(l.q), p])
	rpt := rp.add(l.t.reshape([1, 3]))
	y := mlx.concatenate([
		mlx.ones([n, 1], .float32),
		mlx.zeros([n, 3], .float32),
		mlx.zeros([n, 1], .float32),
		rpt,
	], 1)
	return y.reshape(x.shape())
}

pub fn (mut l MotorGroupLayer) backward(grad mlx.Array) mlx.Array {
	gm := flatten_leading(grad)
	n := gm.dim(0)
	gd := gm.take_axis(mlx.array_i32([i32(5), 6, 7], [3]), 1)
	p := flatten_leading(l.x).take_axis(mlx.array_i32([i32(5), 6, 7], [3]), 1)

	// dt: the dual part is R·P + t, so the cotangent is the translation grad.
	l.dt = gd.sum_axis(0, false)

	// dq: chain through the normalised quaternion.
	r := l.q
	s := mlx.s_add(r.square().sum(), 1e-12).sqrt().item_f32()
	s2 := s * s
	qn := r.divide(mlx.f32_scalar(s))
	w := qn.take_axis(mlx.array_i32([i32(0)], [1]), 0) // [1]
	v := qn.take_axis(mlx.array_i32([i32(1), 2, 3], [3]), 0) // [3]
	v_b := v.reshape([1, 3]).broadcast_to([n, 3])
	v_cross := cross3(v_b, p)
	dot_vp := p.multiply(v_b).sum_axis(1, true) // [n,1]
	// JA[j,i] = delta_ji (v·P) + v_j P_i - 2 P_j v_i   (d(v×(v×P))/dv)
	id3 := mlx.array_f32(identity3(), [3, 3])
	dot3 := dot_vp.reshape([n, 1, 1]).broadcast_to([n, 3, 3]).multiply(id3.reshape([1, 3, 3]).broadcast_to([
		n,
		3,
		3,
	]))
	outer_vp := mlx.einsum('nj,ni->nji', [v_b, p])
	outer_pv := mlx.einsum('nj,ni->nji', [p, v_b])
	ja := dot3.add(outer_vp).subtract(outer_pv.multiply(mlx.f32_scalar(2.0)))
	// Jcross[j,i] = eps[j,i,k] P_k   (d(v×P)/dv)
	jcross := mlx.einsum('nk,jik->nji', [p, mlx.array_f32(eps_tensor(), [3, 3, 3])])
	jrp_v := ja.multiply(mlx.f32_scalar(2.0)).add(jcross.multiply(w.multiply(mlx.f32_scalar(2.0))))
	jrp_w := v_cross.multiply(mlx.f32_scalar(2.0)) // [n, 3]
	// dv/dr = (I34 - rv⊗r / s²) / s ; dw/dr = (e0 - r0·r / s²) / s
	rv := r.take_axis(mlx.array_i32([i32(1), 2, 3], [3]), 0)
	r0 := r.take_axis(mlx.array_i32([i32(0)], [1]), 0)
	outer_rv_r := mlx.einsum('i,j->ij', [rv, r])
	i34 := mlx.array_f32(identity34(), [3, 4])
	dv_dr := mlx.s_div(i34.subtract(mlx.s_mul(outer_rv_r, 1.0 / s2)), s)
	e0 := mlx.array_f32([f32(1), 0, 0, 0], [1, 4])
	r0r := r.multiply(r0).reshape([1, 4])
	dw_dr := mlx.s_div(e0.subtract(mlx.s_mul(r0r, 1.0 / s2)), s)
	// contract with the cotangent
	t1 := mlx.einsum('nc,ncm,mj->j', [gd, jrp_v, dv_dr])
	t2 := mlx.einsum('nc,nc,sj->j', [gd, jrp_w, dw_dr])
	l.dq = t1.add(t2)

	// dx: only the dual-vector slot carries signal; dy/dP = R.
	m := rot_matrix(l.q)
	dxp := mlx.einsum('nc,ci->ni', [gd, m])
	dx := mlx.concatenate([
		mlx.zeros([n, 1], .float32),
		mlx.zeros([n, 3], .float32),
		mlx.zeros([n, 1], .float32),
		dxp,
	], 1)
	return dx.reshape(l.x.shape())
}

pub fn (mut l MotorGroupLayer) params() []mlx.Array {
	return [l.q, l.t]
}

pub fn (mut l MotorGroupLayer) grads() []mlx.Array {
	return [l.dq, l.dt]
}

pub fn (mut l MotorGroupLayer) set_params(ps []mlx.Array) {
	l.q = ps[0]
	l.t = ps[1]
}

pub fn (mut l MotorGroupLayer) set_training(training bool) {}

pub fn (mut l MotorGroupLayer) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.q', l.q)
	m.insert('${prefix}.t', l.t)
}

pub fn (mut l MotorGroupLayer) load_params(m mlx.MapStringToArray, prefix string) {
	l.q = reshape_to(m.get('${prefix}.q'), [4], '${prefix}.q')
	l.t = reshape_to(m.get('${prefix}.t'), [3], '${prefix}.t')
	l.q.eval()
	l.t.eval()
}
