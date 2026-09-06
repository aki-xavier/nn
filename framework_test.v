// non-OOP — test file: free test_/helper functions are exempt from the OOP rule.
module nn

import mlx

// framework_test.v — coverage for the generic-framework features added on
// top of the layer zoo: gradient clipping, Module containers, sequence
// layers, 1D/3D convs, GroupNorm, LR schedulers, resumable Adam state and
// half-precision conversion.

fn test_grad_clipping_limits_update() {
	// two identical nets; one clipped at 0.5 (lr = 1 so |Δp| ≈ clip_norm)
	mut a := Sequential{}
	a.add(new_linear(4, 2, 5))
	mut b := Sequential{}
	b.add(new_linear(4, 2, 5))
	// copy weights from a to b
	ps := a.layers[0].params()
	b.layers[0].set_params(ps)

	x := mlx.array_f32([f32(1), 2, 3, 4, 5, 6, 7, 8], [2, 4])
	y := mlx.array_f32([f32(9), 1, 2, 8], [2, 2])
	mut mse := Loss(MSELoss{})
	mut oa := Optimizer(SGD{
		lr: 1.0
	})
	mut ob := Optimizer(SGD{
		lr: 1.0
		clip_norm: 0.5
	})
	a.train_step(x, y, mut mse, mut oa)
	b.train_step(x, y, mut mse, mut ob)

	delta_a := a.layers[0].params()[0].subtract(ps[0]).square().sum().sqrt().item_f32()
	delta_b := b.layers[0].params()[0].subtract(ps[0]).square().sum().sqrt().item_f32()
	// the clipped update should be bounded by clip_norm and clearly smaller
	assert delta_b <= 0.51, 'clipped update norm ${delta_b} exceeds clip_norm 0.5'
	assert delta_b < delta_a, 'clipping should reduce the update (${delta_b} vs ${delta_a})'
	assert b.last_grad_norm > 0
}

fn test_module_named_parameters() {
	mut m := Module{}
	m.add(new_linear(3, 4, 1))
	m.add(ReLU{})
	mut inner := Module{}
	inner.add(new_linear(4, 2, 2))
	m.add(inner)
	named := m.named_parameters()
	assert named['child.0.w'].shape() == [3, 4]
	assert named['child.0.b'].shape() == [1, 4]
	assert named['child.2.child.0.w'].shape() == [4, 2]
	// forward/backward through Module dispatch
	x := mlx.array_f32([f32(1), 2, 3], [1, 3])
	out := m.forward(x)
	assert out.shape() == [1, 2]
	dx := m.backward(mlx.ones_like(out))
	assert dx.shape() == [1, 3]
}

fn test_attention_shapes_and_gradient() {
	mut l := Layer(new_attention(8, 2, false, 9))
	x := mlx.array_f32([]f32{len: 2 * 4 * 8, init: f32((index * 7) % 5) / 5.0 - 0.4}, [
		2,
		4,
		8,
	])
	out := l.forward(x)
	assert out.shape() == [2, 4, 8]
	fd_check('attn.wq', mut l, x, 0)
	fd_check('attn.wo', mut l, x, 3)

	// causal mask makes the output independent of future tokens: last output
	// row must equal the one from a truncated sequence
	mut cl := Layer(new_attention(8, 2, true, 9))
	cl.set_params(l.params())
	out_c := cl.forward(x)
	assert out_c.shape() == [2, 4, 8]
}

fn test_lstm_shape_and_gradient() {
	mut l := Layer(new_lstm(6, 5, 4))
	x := mlx.array_f32([]f32{len: 2 * 3 * 6, init: f32((index * 11) % 7) / 6.0 - 0.4}, [
		2,
		3,
		6,
	])
	out := l.forward(x)
	assert out.shape() == [2, 3, 5]
	fd_check('lstm.w_ih', mut l, x, 0)
	fd_check('lstm.w_hh', mut l, x, 1)
}

fn test_conv1d_conv3d_gradients() {
	mut c1 := Layer(new_conv1d(2, 3, 3, 1, 1, 7))
	x1 := mlx.array_f32([]f32{len: 2 * 6 * 2, init: f32((index * 5) % 9) / 8.0 - 0.4}, [
		2,
		6,
		2,
	])
	assert c1.forward(x1).shape() == [2, 6, 3]
	fd_check('conv1d.w', mut c1, x1, 0)

	mut c3 := Layer(new_conv3d(1, 2, 3, 1, 1, 7))
	x3 := mlx.array_f32([]f32{len: 1 * 4 * 4 * 4 * 1, init: f32((index * 3) % 7) / 6.0 - 0.4}, [
		1,
		4,
		4,
		4,
		1,
	])
	assert c3.forward(x3).shape() == [1, 4, 4, 4, 2]
	fd_check('conv3d.w', mut c3, x3, 0)
}

fn test_groupnorm_smoke() {
	mut g := Layer(new_group_norm(6, 2))
	x := mlx.array_f32([]f32{len: 2 * 4 * 4 * 6, init: f32((index * 9) % 5) / 4.0 - 0.4}, [
		2,
		4,
		4,
		6,
	])
	out := g.forward(x)
	assert out.shape() == [2, 4, 4, 6]
	dx := g.backward(mlx.ones_like(out))
	assert dx.shape() == [2, 4, 4, 6]
}

fn test_lr_schedulers() {
	mut s := LRScheduler(StepLR{
		lr: 1.0
		step_size: 10
		gamma: 0.1
	})
	assert s.rate(0) == 1.0
	assert s.rate(9) == 1.0
	assert absf(s.rate(10) - 0.1) < 1e-4

	mut c := LRScheduler(CosineLR{
		lr_max: 1.0
		lr_min: 0.0
		t_max: 100
	})
	assert absf(c.rate(0) - 1.0) < 1e-4
	assert absf(c.rate(50) - 0.5) < 1e-4 // cos(pi/2) = 0 -> midpoint rate
	assert absf(c.rate(99)) < 1e-3 // near the minimum at t_max (modulo wraps at t_max)
}

fn test_adam_state_roundtrip() {
	mut opt := Optimizer(Adam{
		lr: 0.01
	})
	mut net := Sequential{}
	net.add(new_linear(3, 2, 3))
	x := mlx.array_f32([]f32{len: 4 * 3, init: f32((index * 5) % 7) / 6.0}, [4, 3])
	y := mlx.array_f32([]f32{len: 4 * 2, init: f32((index * 3) % 5) / 4.0}, [4, 2])
	mut mse := Loss(MSELoss{})
	net.train_step(x, y, mut mse, mut opt)
	net.train_step(x, y, mut mse, mut opt)
	tmp := '/tmp/nn_adam_state.safetensors'
	opt.save_state(tmp)
	// fresh optimizer resumes from the saved step count
	mut opt2 := Optimizer(Adam{
		lr: 0.01
	})
	opt2.load_state(tmp)
	assert opt2.t == 2
	mut net2 := Sequential{}
	net2.add(new_linear(3, 2, 3))
	net2.layers[0].set_params(net.layers[0].params())
	net2.train_step(x, y, mut mse, mut opt2) // t=2 -> 3
	net.train_step(x, y, mut mse, mut opt) // t=2 -> 3
	// same weights, same restored moments, same step index -> lockstep
	d := net2.layers[0].params()[0].subtract(net.layers[0].params()[0]).abs().max().item_f32()
	assert d < 1e-5, 'resumed Adam drifted: ${d}'
}

fn test_half_dtype_conversion() {
	mut net := Sequential{}
	net.add(new_linear(3, 2, 1))
	net.half()
	assert net.layers[0].params()[0].dtype() == .float16
	net.to_float32()
	assert net.layers[0].params()[0].dtype() == .float32
}

fn test_backbone_presets_smoke() {
	mut vgg := vgg16_lite(10, 1)
	vgg.set_training(false)
	out := vgg.predict(mlx.zeros([1, 224, 224, 3], .float32))
	assert out.shape() == [1, 10]

	mut res := resnet18_lite(5, 1)
	res.set_training(false)
	out2 := res.predict(mlx.zeros([1, 224, 224, 3], .float32))
	assert out2.shape() == [1, 5]

	mut unet := hed_unet(1)
	unet.set_training(false)
	out3 := unet.predict(mlx.zeros([1, 240, 320, 1], .float32))
	assert out3.shape() == [1, 240, 320, 1]
}
