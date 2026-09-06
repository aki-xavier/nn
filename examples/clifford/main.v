module main

import mlx
import nn

// End-to-end smoke test for the scalar/rotor/motor layers: random motors act
// on a fixed template point, and a small mixed net (CliffordLinear + motor
// GroupLayer + ReprSwitch + readout) learns to predict the transformed
// point.  The trained net is saved and reloaded to show reproducibility.

fn random_motors(n int, seed u64) [][6]f32 {
	mut out := [][6]f32{len: n}
	key := mlx.random_key(seed)
	rot := mlx.random_uniform(mlx.f32_scalar(-1.0), mlx.f32_scalar(1.0), [n, 3], .float32, key)
	trans := mlx.random_uniform(mlx.f32_scalar(-0.5), mlx.f32_scalar(0.5), [n, 3], .float32, key)
	rd := rot.data_f32()
	td := trans.data_f32()
	for i in 0 .. n {
		out[i][0] = rd[3 * i]
		out[i][1] = rd[3 * i + 1]
		out[i][2] = rd[3 * i + 2]
		out[i][3] = td[3 * i]
		out[i][4] = td[3 * i + 1]
		out[i][5] = td[3 * i + 2]
	}
	return out
}

// motor_action computes 1 + ε·(R·P + t) for the template point P = x̂.
fn motor_action(params [6]f32) mlx.Array {
	mut gm := nn.new_group_layer(.motor)
	gm.set_params([
		mlx.array_f32([params[0], params[1], params[2]], [3]),
		mlx.array_f32([params[3], params[4], params[5]], [3]),
	])
	pt := mlx.array_f32([f32(1), 0, 0, 0, 0, 1, 0, 0], [1, 8])
	return gm.forward(pt)
}

fn main() {
	println('mlx ${mlx.version()}  gpu: ${mlx.gpu_available()}')
	n := 64
	ms := random_motors(n, 7)

	mut xs := []mlx.Array{}
	mut ys := []mlx.Array{}
	for i in 0 .. n {
		out := motor_action(ms[i])
		xs << out
		// target: transformed point as 3-d coords (dual part 5..7)
		ys << mlx.array_f32(out.data_f32()[5..8], [1, 3])
	}
	ds := nn.Dataset{
		x: mlx.concatenate(xs, 0).reshape([n, 1, 8])
		y: mlx.concatenate(ys, 0)
	}

	// clifford(2 channels) -> motor group layer -> flatten -> head
	// (ReprSwitch semantics are unit-tested in clifford_test.v; here the
	// information-carrying dual components must stay in the main path.)
	mut net := nn.Sequential{}
	net.add(nn.new_clifford_linear(.motor, 1, 2, 31))
	net.add(nn.new_motor_group_layer())
	net.add(nn.Flatten{})
	net.add(nn.new_linear(16, 3, 32))

	mut dl := nn.new_dataloader(ds, 16, true)
	mut mse := nn.Loss(nn.MSELoss{})
	mut opt := nn.Optimizer(nn.Adam{
		lr: 0.01
	})
	net.fit_loader(mut dl, mut mse, mut opt, 60, 10)

	net.set_training(false)
	pred := net.predict(ds.x)
	err := pred.subtract(ds.y).abs().max().item_f32()
	println('final max |pred - target| = ${err:.5f}')

	net.save('clifford_demo.safetensors')
	mut net2 := nn.Sequential{}
	net2.add(nn.new_clifford_linear(.motor, 1, 2, 0))
	net2.add(nn.new_motor_group_layer())
	net2.add(nn.Flatten{})
	net2.add(nn.new_linear(16, 3, 0))
	net2.set_training(false)
	net2.load('clifford_demo.safetensors')
	pred2 := net2.predict(ds.x)
	diff := pred.subtract(pred2).abs().max().item_f32()
	println('reload max |pred - pred2| = ${diff:.3e}')
	if diff < 1e-5 {
		println('PASS: save/load roundtrip')
	} else {
		println('FAIL: weights diverge')
	}
}
