// non-OOP — test file: free test_/helper functions are exempt from the OOP rule.
module nn

import mlx

// motor_test.v — MotorGroupLayer (hard-wired SE(3)) validation: forward
// equivalence with the generic vjp GroupLayer, finite-difference gradient
// checks on the analytic backward, and rigid-motion geometry.

fn test_motor_forward_matches_vjp_grouplayer() {
	// same rigid motion through both implementations:
	// rotvec (0,0,pi/2) equivalence quaternion [cos(pi/4), 0, 0, sin(pi/4)]
	// and translation (1, 0, 0) applied to point P = x̂.
	mut gen := Layer(new_group_layer(.motor))
	gen.set_params([mlx.array_f32([f32(0), 0, 1.5707963], [3]), mlx.array_f32([f32(1), 0, 0], [3])])
	mut spec := Layer(new_motor_group_layer())
	spec.set_params([mlx.array_f32([f32(0.7071068), 0, 0, 0.7071068], [4]),
		mlx.array_f32([f32(1), 0, 0], [3])])

	pt := mlx.array_f32([f32(1), 0, 0, 0, 0, 1, 0, 0], [1, 8])
	v1 := gen.forward(pt).data_f32()
	v2 := spec.forward(pt).data_f32()
	// 1 + ε·(R x̂ + t) = 1 + ε·(0, 1, 1, 0)
	assert absf(v2[5] - 1.0) < 1e-3 && absf(v2[6] - 1.0) < 1e-3, 'motor geometry: ${v2}'
	mut dmax := f32(0)
	for i in 0 .. 8 {
		d := absf(v1[i] - v2[i])
		if d > dmax {
			dmax = d
		}
	}
	assert dmax < 1e-4, 'generic vs hard-wired mismatch ${dmax}'
}

fn test_motor_analytic_gradient() {
	mut l := Layer(new_motor_group_layer())
	// random-ish points as inputs (dual vector slots)
	pts := mlx.array_f32([]f32{len: 3 * 8, init: f32((index * 13) % 9) / 8.0 - 0.4}, [
		3,
		8,
	])
	fd_check('motor.q identity', mut l, pts, 0)
	fd_check('motor.t identity', mut l, pts, 1)
	// also verify at a non-trivial rotation + translation
	l.set_params([
		mlx.array_f32([f32(0.7), 0.2, -0.3, 0.5], [4]),
		mlx.array_f32([f32(0.3), -0.4, 0.2], [3]),
	])
	fd_check('motor.q random', mut l, pts, 0)
	fd_check('motor.t random', mut l, pts, 1)
}

fn test_motor_rotation_geometry() {
	// q = 90° about z, no translation: P = x̂ -> ŷ
	mut spec := Layer(new_motor_group_layer())
	spec.set_params([mlx.array_f32([f32(0.7071068), 0, 0, 0.7071068], [4]), mlx.zeros([3], .float32)])
	pt := mlx.array_f32([f32(1), 0, 0, 0, 0, 1, 0, 0], [1, 8])
	v := spec.forward(pt).data_f32()
	assert absf(v[5]) < 1e-4 && absf(v[6] - 1.0) < 1e-4, 'rotate x̂ -> ŷ: ${v}'
}
