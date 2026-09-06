module main

import mlx
import nn

// Comparison: bounded-CGA-group-hybrid vs fully-free layers at matched
// parameter counts on the conformal-inversion task.  We measure (a) final
// inversion accuracy and (b) training gradient-norm stability (max/mean over
// epochs) — the property the bounded group layer is designed to buy.

const n = 64
const epochs = 60

fn absf32(v f32) f32 {
	if v < 0 {
		return -v
	}
	return v
}

fn rand_vec(lo f32, hi f32, seed u64, shape []int) []f32 {
	key := mlx.random_key(seed)
	u := mlx.random_uniform(mlx.f32_scalar(lo), mlx.f32_scalar(hi), shape, .float32, key)
	return u.data_f32()
}

// make_dataset builds the same conformal-inversion task as examples/cga.
fn make_dataset() nn.Dataset {
	template := mlx.array_f32([f32(0.5), 0.2, -0.4, -0.7, 0.8, 0.3, 0.1, -0.5, 0.9], [
		3,
		3,
	])
	mut xs := []mlx.Array{}
	mut ys := []mlx.Array{}
	for i in 0 .. n {
		rot := rand_vec(-1.4, 1.4, 100 + u64(i), [3])
		tr := rand_vec(-0.8, 0.8, 200 + u64(i), [3])
		sc := 0.4 + 1.6 * absf32(rand_vec(-1.0, 1.0, 300 + u64(i), [1])[0])
		mut l1 := nn.new_cga_group_layer(8.0)
		l1.set_params([nn.cga_rotation_params(rot, 0.9, 1.0)])
		mut l2 := nn.new_cga_group_layer(8.0)
		l2.set_params([nn.cga_translation_params(tr, 0.5)])
		mut l3 := nn.new_cga_group_layer(8.0)
		l3.set_params([nn.cga_dilation_params(sc, -1.0)])
		mut f := nn.conformal_point_pub(template)
		f = nn.Layer(l1).forward(f)
		f = nn.Layer(l2).forward(f)
		f = nn.Layer(l3).forward(f)
		xs << f.reshape([1, 3, 32])
		ys << template.reshape([1, 9])
	}
	return nn.Dataset{
		x: mlx.concatenate(xs, 0)
		y: mlx.concatenate(ys, 0)
	}
}

// net_a: bounded CGA group hybrid (CliffordLinear + CGAGroupLayer).
fn net_a(seed u64) nn.Sequential {
	mut net := nn.Sequential{}
	net.add(nn.new_clifford_linear(.cga, 3, 2, seed))
	net.add(nn.new_cga_group_layer(8.0))
	net.add(nn.Flatten{})
	net.add(nn.new_linear(64, 9, seed + 1))
	return net
}

// net_b: fully free layers at a matched parameter budget
// (CliffordLinear(3->2) + (2->2) + (2->1) + head ≈ 841 params vs 851).
fn net_b(seed u64) nn.Sequential {
	mut net := nn.Sequential{}
	net.add(nn.new_clifford_linear(.cga, 3, 2, seed))
	net.add(nn.new_clifford_linear(.cga, 2, 2, seed + 1))
	net.add(nn.new_clifford_linear(.cga, 2, 1, seed + 2))
	net.add(nn.Flatten{})
	net.add(nn.new_linear(32, 9, seed + 3))
	return net
}

fn run_net(mut net nn.Sequential, ds nn.Dataset, tag string) {
	nparams := net.param_count()
	println('== ${tag}  params=${nparams}')
	mut dl := nn.new_dataloader(ds, 16, true)
	mut mse := nn.Loss(nn.MSELoss{})
	mut opt := nn.Optimizer(nn.Adam{
		lr: 0.003
		clip_norm: 5.0 // same safety net for both
	})
	mut max_grad := f32(0)
	mut sum_grad := f32(0)
	for e in 1 .. epochs + 1 {
		dl.reset()
		for {
			batch := dl.next() or { break }
			net.train_step(batch.x, batch.y, mut mse, mut opt)
		}
		if net.last_grad_norm > max_grad {
			max_grad = net.last_grad_norm
		}
		sum_grad += net.last_grad_norm
	}
	net.set_training(false)
	pred := net.predict(ds.x)
	err := pred.subtract(ds.y).abs().max().item_f32()
	println('  final max err = ${err:.5f}   max|grad| = ${max_grad:.3f}   mean|grad| = ${sum_grad / f32(epochs):.3f}')
}

fn main() {
	println('mlx ${mlx.version()}  gpu: ${mlx.gpu_available()}')
	ds := make_dataset()
	for seed in u64(1) .. u64(5) {
		mut a := net_a(seed * 100)
		run_net(mut a, ds, 'A(group-hybrid, seed ${seed})')
		mut b := net_b(seed * 100)
		run_net(mut b, ds, 'B(free,        seed ${seed})')
	}
}
