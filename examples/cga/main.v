module main

import mlx
import nn

// Conformal-inversion toy: random conformal transforms (bounded rotation +
// translation + dilation) act on a fixed template point set; a net mixing a
// free CGA CliffordLinear with a CGAGroupLayer learns to recover the
// canonical points from the transformed ones.  Demonstrates the conformal
// group layer end to end (training, extraction, save/load).

fn rand_vec(n int, lo f32, hi f32, seed u64, shape []int) []f32 {
	key := mlx.random_key(seed)
	u := mlx.random_uniform(mlx.f32_scalar(lo), mlx.f32_scalar(hi), shape, .float32, key)
	return u.data_f32()
}

// transform_params generates one conformal parameter triple.
fn transform_params(i int) (nn.Layer, nn.Layer, nn.Layer) {
	rot := rand_vec(3, -1.0, 1.0, 100 + u64(i), [3])
	tr := rand_vec(3, -0.4, 0.4, 200 + u64(i), [3])
	sc := 0.6 + 0.5 * absf32(rand_vec(1, -1.0, 1.0, 300 + u64(i), [1])[0])
	mut l1 := nn.new_cga_group_layer(8.0)
	l1.set_params([nn.cga_rotation_params(rot, 0.9, 1.0)])
	mut l2 := nn.new_cga_group_layer(8.0)
	l2.set_params([nn.cga_translation_params(tr, 0.5)])
	mut l3 := nn.new_cga_group_layer(8.0)
	l3.set_params([nn.cga_dilation_params(sc, -1.0)])
	return nn.Layer(l1), nn.Layer(l2), nn.Layer(l3)
}

fn absf32(v f32) f32 {
	if v < 0 {
		return -v
	}
	return v
}

fn main() {
	println('mlx ${mlx.version()}  gpu: ${mlx.gpu_available()}')
	// template: three fixed points
	template := mlx.array_f32([f32(0.5), 0.2, -0.4, -0.7, 0.8, 0.3, 0.1, -0.5, 0.9], [
		3,
		3,
	])

	n := 64
	mut xs := []mlx.Array{}
	mut ys := []mlx.Array{}
	for i in 0 .. n {
		l1, l2, l3 := transform_params(i)
		pc := nn.conformal_point_pub(template)
		mut f := l1.forward(pc)
		f = l2.forward(f)
		f = l3.forward(f)
		// f is (3, 32) (three conformal points); keep them as one sample of
		// three channels
		xs << f.reshape([1, 3, 32])
		ys << template.reshape([1, 9])
	}
	ds := nn.Dataset{
		x: mlx.concatenate(xs, 0)
		y: mlx.concatenate(ys, 0)
	}

	mut net := nn.Sequential{}
	net.add(nn.new_clifford_linear(.cga, 3, 2, 31))
	net.add(nn.new_cga_group_layer(8.0))
	net.add(nn.Flatten{})
	net.add(nn.new_linear(64, 9, 32))

	println('ds.x ${ds.x.shape()} ds.y ${ds.y.shape()}')
	mut dl := nn.new_dataloader(ds, 16, true)
	mut mse := nn.Loss(nn.MSELoss{})
	mut opt := nn.Optimizer(nn.Adam{
		lr: 0.003
	})
	net.fit_loader(mut dl, mut mse, mut opt, 60, 10)

	net.set_training(false)
	pred := net.predict(ds.x)
	err := pred.subtract(ds.y).abs().max().item_f32()
	println('final max |pred - canonical| = ${err:.5f}')

	net.save('cga_demo.safetensors')
	mut net2 := nn.Sequential{}
	net2.add(nn.new_clifford_linear(.cga, 3, 2, 0))
	net2.add(nn.new_cga_group_layer(8.0))
	net2.add(nn.Flatten{})
	net2.add(nn.new_linear(64, 9, 0))
	net2.set_training(false)
	net2.load('cga_demo.safetensors')
	pred2 := net2.predict(ds.x)
	diff := pred.subtract(pred2).abs().max().item_f32()
	if diff < 1e-5 {
		println('PASS: conformal save/load roundtrip')
	} else {
		println('FAIL: weights diverge')
	}
}
