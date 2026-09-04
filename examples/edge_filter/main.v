module main

import mlx
import nn

// Learn edge detection from synthetic data: generate piecewise-constant
// images, compute ground-truth edges with fixed Sobel kernels, and train a
// tiny CNN (conv/pool/upsample/residual) to reproduce them.

// sobel_kernel returns a fixed 3x3 Sobel filter as a conv weight [1, 3, 3, 1].
fn sobel_kernel(vertical bool) mlx.Array {
	if vertical {
		return mlx.array_f32([f32(-1), 0, 1, -2, 0, 2, -1, 0, 1], [1, 3, 3, 1])
	}
	return mlx.array_f32([f32(-1), -2, -1, 0, 0, 0, 1, 2, 1], [1, 3, 3, 1])
}

// make_dataset builds n piecewise-constant 16x16 images (4x4 random blocks
// upsampled 4x) plus binary edge targets from Sobel magnitude > 1.0.
fn make_dataset(n int, seed u64) nn.Dataset {
	key := mlx.random_key(seed)
	zero := mlx.f32_scalar(0.0)
	one := mlx.f32_scalar(1.0)
	blocks := mlx.random_uniform(zero, one, [n, 4, 4, 1], .float32, key)
	images := blocks.reshape([n, 4, 1, 4, 1, 1]).broadcast_to([n, 4, 4, 4, 4, 1]).reshape([
		n,
		16,
		16,
		1,
	])
	gx := mlx.conv2d(images, sobel_kernel(true), 1, 1, 1)
	gy := mlx.conv2d(images, sobel_kernel(false), 1, 1, 1)
	mag := gx.abs().add(gy.abs())
	edges := mlx.s_gt(mag, 1.0).astype(.float32)
	return nn.Dataset{
		x: images
		y: edges
	}
}

fn main() {
	println('mlx ${mlx.version()}  gpu: ${mlx.gpu_available()}')
	ds := make_dataset(64, 7)
	println('dataset: x=${ds.x.shape()} y=${ds.y.shape()}  edge fraction=${ds.y.mean().item_f32():.3f}')

	// train/test split: first 48 / last 16
	lo := mlx.arange(0, 48, 1, .int32)
	hi := mlx.arange(48, 64, 1, .int32)
	train := nn.Dataset{
		x: ds.x.take_axis(lo, 0)
		y: ds.y.take_axis(lo, 0)
	}
	test := nn.Dataset{
		x: ds.x.take_axis(hi, 0)
		y: ds.y.take_axis(hi, 0)
	}

	// conv -> relu -> residual -> pool -> conv -> upsample -> conv -> sigmoid
	mut net := nn.Sequential{}
	net.add(nn.new_conv2d(1, 8, 3, 1, 1, 11))
	net.add(nn.ReLU{})
	net.add(nn.new_residual([nn.Layer(nn.new_conv2d(8, 8, 3, 1, 1, 12)), nn.Layer(nn.ReLU{})]))
	net.add(nn.new_max_pool2d(2))
	net.add(nn.new_conv2d(8, 16, 3, 1, 1, 13))
	net.add(nn.ReLU{})
	net.add(nn.new_upsample2d(2))
	net.add(nn.new_conv2d(16, 8, 3, 1, 1, 14))
	net.add(nn.ReLU{})
	net.add(nn.new_conv2d(8, 1, 3, 1, 1, 15))
	net.add(nn.Sigmoid{})

	net.set_training(false)
	pred0 := net.predict(test.x)
	println('before training: ${nn.edge_metrics(pred0, test.y)}')

	mut dl := nn.new_dataloader(train, 16, true)
	mut criterion := nn.Loss(nn.WeightedBCELoss{
		w_pos: 2.0
	})
	mut opt := nn.Optimizer(nn.Adam{
		lr: 0.01
	})
	net.fit_loader(mut dl, mut criterion, mut opt, 200, 40)

	net.set_training(false)
	pred1 := net.predict(test.x)
	println('after training:  ${nn.edge_metrics(pred1, test.y)}')
}
