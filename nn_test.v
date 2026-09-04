// non-OOP — test file: free test_/helper functions are exempt from the OOP rule.
module nn

import mlx

// nn_test.v — gradient and shape smoke tests.  Conv2d/Linear backward are
// checked against finite differences; pooling/containers/norm/dropout are
// checked for gradient shape and mass conservation.

fn absf(v f32) f32 {
	if v < 0 {
		return -v
	}
	return v
}

// set_param_f32 replaces parameter idx of layer l with flat f32 values.
fn set_param_f32(mut l Layer, idx int, vals []f32) {
	mut ps := l.params()
	shape := ps[idx].shape()
	ps[idx] = mlx.array_f32(vals, shape)
	l.set_params(ps)
}

// fd_check compares analytic backward gradients with finite differences on
// parameter idx, for an all-ones cotangent.
fn fd_check(name string, mut l Layer, x mlx.Array, idx int) {
	out := l.forward(x)
	g := mlx.ones_like(out)
	l.backward(g)
	analytic := l.grads()[idx].data_f32()

	base := l.params()[idx].data_f32()
	eps := f32(1e-3)
	mut bad := 0
	mut ncheck := analytic.len
	if ncheck > 8 {
		ncheck = 8
	}
	for i in 0 .. ncheck {
		mut plus := base.clone()
		plus[i] = base[i] + eps
		mut minus := base.clone()
		minus[i] = base[i] - eps
		set_param_f32(mut l, idx, plus)
		hi := l.forward(x).sum().item_f32()
		set_param_f32(mut l, idx, minus)
		lo := l.forward(x).sum().item_f32()
		numeric := (hi - lo) / (2 * eps)
		if absf(analytic[i] - numeric) > 0.05 * (absf(numeric) + 1.0) {
			bad++
			eprintln('${name} param ${idx} elem ${i}: analytic=${analytic[i]:.5f} numeric=${numeric:.5f}')
		}
	}
	set_param_f32(mut l, idx, base)
	assert bad == 0, '${name}: ${bad} mismatched gradient elements'
}

fn test_conv2d_gradient() {
	mut l := Layer(new_conv2d(2, 3, 3, 1, 1, 99))
	mut xv := []f32{len: 2 * 4 * 4 * 2}
	for i in 0 .. xv.len {
		xv[i] = f32((i * 37) % 11) / 10.0 - 0.5
	}
	x := mlx.array_f32(xv, [2, 4, 4, 2])
	fd_check('conv2d.w', mut l, x, 0)
	fd_check('conv2d.b', mut l, x, 1)
}

fn test_linear_gradient() {
	mut l := Layer(new_linear(4, 3, 99))
	x := mlx.array_f32([f32(0.5), -0.3, 0.8, 0.1, 1.2, -0.7, 0.4, 0.9], [2, 4])
	fd_check('linear.w', mut l, x, 0)
	fd_check('linear.b', mut l, x, 1)
}

fn test_maxpool_backward_shape() {
	mut l := Layer(new_max_pool2d(2))
	x := mlx.array_f32([]f32{len: 2 * 4 * 4 * 3, init: f32((index * 13) % 7) / 6.0}, [
		2,
		4,
		4,
		3,
	])
	out := l.forward(x)
	assert out.shape() == [2, 2, 2, 3]
	dx := l.backward(mlx.ones_like(out))
	assert dx.shape() == [2, 4, 4, 3]
	// each window's gradient sums to 1 (mass conservation)
	err := dx.reshape([2, 2, 2, 2, 2, 3]).sum_axes([2, 4], false).subtract(mlx.ones([
		2,
		2,
		2,
		3,
	], .float32)).abs().max().item_f32()
	assert err < 1e-5
}

fn test_container_shapes() {
	mut res := Layer(new_residual([Layer(new_conv2d(4, 4, 3, 1, 1, 5)), Layer(ReLU{})]))
	x := mlx.zeros([1, 8, 8, 4], .float32)
	out := res.forward(x)
	assert out.shape() == [1, 8, 8, 4]
	dx := res.backward(mlx.ones_like(out))
	assert dx.shape() == [1, 8, 8, 4]

	mut sk := Layer(new_skip([Layer(new_conv2d(4, 6, 3, 1, 1, 6)), Layer(ReLU{})]))
	out2 := sk.forward(x)
	assert out2.shape() == [1, 8, 8, 10]
	dx2 := sk.backward(mlx.ones_like(out2))
	assert dx2.shape() == [1, 8, 8, 4]
}

fn test_checkpoint_loading() {
	// fabricate a torch-convention checkpoint: conv 2->3, weight NCHW [3,2,3,3]
	w_torch := mlx.array_f32([]f32{len: 3 * 2 * 3 * 3, init: f32((index * 7) % 13) / 12.0 - 0.5}, [
		3,
		2,
		3,
		3,
	])
	b := mlx.array_f32([f32(0.1), -0.2, 0.3], [3])
	m := mlx.new_map_string_to_array()
	m.insert('features.0.weight', w_torch)
	m.insert('features.0.bias', b)
	meta := mlx.new_map_string_to_string()
	path := '/tmp/nn_test_ckpt.safetensors'
	mlx.save_safetensors(path, m, meta)
	m.free()
	meta.free()

	mut ckpt := open_checkpoint(path)
	defer {
		ckpt.close()
	}
	assert ckpt.has('features.0.weight')
	assert !ckpt.has('nope')
	assert ckpt.shape_of('features.0.weight') == [3, 2, 3, 3]
	assert ckpt.keys().len == 2

	// perm converts NCHW -> NHWC-weight layout
	t := ckpt.tensor('features.0.weight', [0, 2, 3, 1])
	assert t.shape() == [3, 3, 3, 2]
	// t[o, i, j, ci] == w_torch[o, ci, i, j]
	assert t.data_f32()[0] == w_torch.data_f32()[0]

	mut net := Sequential{}
	net.add(new_conv2d(2, 3, 3, 1, 1, 0))
	net.load_checkpoint(ckpt, [
		torch_conv_rule('features.0.weight', 'layers.0.w'),
		plain_rule('features.0.bias', 'layers.0.b'),
	])
	// reference: conv with manually permuted weight
	x := mlx.array_f32([]f32{len: 1 * 5 * 5 * 2, init: f32((index * 11) % 7) / 6.0}, [
		1,
		5,
		5,
		2,
	])
	w_ref := w_torch.transpose_axes([0, 2, 3, 1])
	ref := mlx.conv2d(x, w_ref, 1, 1, 1).add(b.reshape([1, 1, 1, 3]))
	pred := net.predict(x)
	assert pred.subtract(ref).abs().max().item_f32() < 1e-5
}

fn test_norm_dropout_smoke() {
	mut ln := Layer(new_layer_norm(4))
	x := mlx.array_f32([]f32{len: 2 * 2 * 2 * 4, init: f32((index * 17) % 9) / 8.0}, [
		2,
		2,
		2,
		4,
	])
	out := ln.forward(x)
	assert out.shape() == [2, 2, 2, 4]
	dx := ln.backward(mlx.ones_like(out))
	assert dx.shape() == [2, 2, 2, 4]

	mut bn := Layer(new_batch_norm2d(4))
	out2 := bn.forward(x)
	assert out2.shape() == [2, 2, 2, 4]
	dx2 := bn.backward(mlx.ones_like(out2))
	assert dx2.shape() == [2, 2, 2, 4]
	bn.set_training(false)
	assert bn.forward(x).shape() == [2, 2, 2, 4]

	mut dr := Layer(new_dropout(0.5))
	assert dr.forward(x).shape() == [2, 2, 2, 4]
	dr.set_training(false)
	// inference mode is the identity
	assert dr.forward(x).subtract(x).abs().max().item_f32() == 0.0
}
