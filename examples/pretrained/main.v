module main

import mlx
import nn

// Simulate loading a pretrained PyTorch-style checkpoint: fabricate a
// safetensors file with torch conventions (NCHW conv weights [out, in, k, k],
// `features.{i}.weight` names), load it into an NHWC nn network via
// LoadRules, and verify the prediction matches a reference computation.

fn main() {
	// --- fabricate a torch-convention checkpoint -------------------------
	// conv1: 1->4 3x3, conv2: 4->2 3x3, weights stored NCHW like PyTorch.
	w1_torch := mlx.random_normal([4, 1, 3, 3], .float32, 0.0, 0.5, mlx.random_key(101))
	b1 := mlx.random_normal([4], .float32, 0.0, 0.1, mlx.random_key(102))
	w2_torch := mlx.random_normal([2, 4, 3, 3], .float32, 0.0, 0.5, mlx.random_key(103))
	b2 := mlx.random_normal([2], .float32, 0.0, 0.1, mlx.random_key(104))

	tensors := mlx.new_map_string_to_array()
	tensors.insert('features.0.weight', w1_torch)
	tensors.insert('features.0.bias', b1)
	tensors.insert('features.2.weight', w2_torch)
	tensors.insert('features.2.bias', b2)
	meta := mlx.new_map_string_to_string()
	mlx.save_safetensors('torch_style.safetensors', tensors, meta)
	println('fabricated torch-style checkpoint')

	// --- inspect the checkpoint header -----------------------------------
	mut ckpt := nn.open_checkpoint('torch_style.safetensors')
	defer {
		ckpt.close()
	}
	println('checkpoint tensors: ${ckpt.keys()}')
	println('features.0.weight declared shape: ${ckpt.shape_of('features.0.weight')} (torch NCHW)')

	// --- build the nn network (NHWC conventions) --------------------------
	mut net := nn.Sequential{}
	net.add(nn.new_conv2d(1, 4, 3, 1, 1, 0))
	net.add(nn.ReLU{})
	net.add(nn.new_conv2d(4, 2, 3, 1, 1, 0))

	// --- load with name mapping + layout conversion -----------------------
	rules := [
		nn.torch_conv_rule('features.0.weight', 'layers.0.w'),
		nn.plain_rule('features.0.bias', 'layers.0.b'),
		nn.torch_conv_rule('features.2.weight', 'layers.2.w'),
		nn.plain_rule('features.2.bias', 'layers.2.b'),
	]
	net.load_checkpoint(ckpt, rules)
	println('loaded into nn network with perm [0, 2, 3, 1]')

	// --- reference computation with raw mlx ops ---------------------------
	x := mlx.random_uniform(mlx.f32_scalar(0.0), mlx.f32_scalar(1.0), [2, 8, 8, 1], .float32, mlx.random_key(7))
	w1 := w1_torch.transpose_axes([0, 2, 3, 1])
	w2 := w2_torch.transpose_axes([0, 2, 3, 1])
	ref := mlx.conv2d(x, w1, 1, 1, 1).add(b1.reshape([1, 1, 1, 4])).maximum(mlx.f32_scalar(0.0))
	ref2 := mlx.conv2d(ref, w2, 1, 1, 1).add(b2.reshape([1, 1, 1, 2]))

	pred := net.predict(x)
	diff := pred.subtract(ref2).abs().max().item_f32()
	println('max |predict - reference| = ${diff:.3e}')
	if diff < 1e-5 {
		println('PASS: checkpoint loading is bit-compatible with the reference')
	} else {
		panic('FAIL: outputs diverge')
	}

	// --- side outputs (HED/FPN-style intermediate taps) -------------------
	taps := net.forward_taps(x, [0, 2])
	println('tap after conv1: ${taps[0].shape()}, tap after conv2: ${taps[1].shape()}')
}
