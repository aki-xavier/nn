module main

import os
import mlx
import nn

// HED-style edge likelihood estimation on BSDS500: a two-level U-Net built
// from Skip containers, trained with class-weighted BCE against the mean of
// the human annotators' boundary maps (soft edge likelihoods).

const img_h = 240
const img_w = 320

// load_split loads `limit` images and their edge-likelihood targets for a
// BSDS split, resized to a fixed [1, 240, 320, c] so mini-batching works.
// Horizontal-flip augmentation is applied only when `augment` is set (train).
fn load_split(split string, limit int, augment bool) nn.Dataset {
	img_dir := 'data/BSDS500-master/BSDS500/data/images/${split}'
	gt_dir := 'data/gt_npy/${split}'
	names := os.ls(img_dir) or { panic(err) }.filter(it.ends_with('.jpg'))
	mut xs := []mlx.Array{}
	mut ys := []mlx.Array{}
	mut n := 0
	for name in names {
		if limit > 0 && n >= limit {
			break
		}
		id := name.all_before_last('.')
		img := nn.load_image('${img_dir}/${name}', 3) or { panic(err) }
		// BSDS500 mixes landscape (321x481) and portrait (481x321); resize to
		// a fixed size so mini-batching works.
		x := nn.resize_nearest(img.tensor, img_h, img_w)
		gt := mlx.load('${gt_dir}/${id}.npy')
		gt4 := gt.reshape([1, gt.dim(0), gt.dim(1), 1])
		y := nn.resize_nearest(gt4, img_h, img_w)
		xs << x
		ys << y
		if augment {
			// horizontal-flip augmentation doubles the training data
			xs << nn.flip_horizontal(x)
			ys << nn.flip_horizontal(y)
		}
		n++
	}
	return nn.Dataset{
		x: mlx.concatenate(xs, 0)
		y: mlx.concatenate(ys, 0)
	}
}

// build_unet builds the two-level U-Net: conv -> Skip(down conv Skip(down) up) -> head.
fn build_unet() nn.Sequential {
	mut net := nn.Sequential{}
	net.add(nn.new_conv2d(3, 32, 3, 1, 1, 21))
	net.add(nn.ReLU{})
	net.add(nn.new_skip([
		nn.Layer(nn.new_max_pool2d(2)),
		nn.Layer(nn.new_conv2d(32, 64, 3, 1, 1, 22)),
		nn.Layer(nn.ReLU{}),
		nn.Layer(nn.new_skip([
			nn.Layer(nn.new_max_pool2d(2)),
			nn.Layer(nn.new_conv2d(64, 128, 3, 1, 1, 23)),
			nn.Layer(nn.ReLU{}),
			nn.Layer(nn.new_upsample2d(2)),
			nn.Layer(nn.new_conv2d(128, 64, 3, 1, 1, 24)),
		])),
		nn.Layer(nn.ReLU{}),
		nn.Layer(nn.new_upsample2d(2)),
		nn.Layer(nn.new_conv2d(128, 32, 3, 1, 1, 25)),
	]))
	net.add(nn.ReLU{})
	net.add(nn.new_conv2d(64, 32, 3, 1, 1, 26))
	net.add(nn.ReLU{})
	net.add(nn.new_conv2d(32, 1, 1, 1, 0, 27))
	net.add(nn.Sigmoid{})
	return net
}

fn main() {
	println('mlx ${mlx.version()}  gpu: ${mlx.gpu_available()}')
	println('loading BSDS500 train split...')
	train := load_split('train', 0, true)
	val := load_split('val', 24, false)
	println('train: ${train.x.shape()}  val: ${val.x.shape()}  edge fraction=${train.y.mean().item_f32():.4f}')

	mut net := build_unet()

	net.set_training(false)
	println('val before training: ${nn.edge_metrics(net.predict(val.x), val.y)}')

	mut dl := nn.new_dataloader(train, 4, true)
	mut criterion := nn.Loss(nn.WeightedBCELoss{
		w_pos: 12.0
	})
	mut opt := nn.Optimizer(nn.Adam{
		lr: 0.003
	})
	net.fit_loader(mut dl, mut criterion, mut opt, 40, 5)

	net.set_training(false)
	pred := net.predict(val.x)
	println('val after training:  ${nn.edge_metrics(pred, val.y)}')

	// save a few predictions for visual inspection
	os.mkdir_all('data/predictions') or {}
	for i in 0 .. 4 {
		sl := mlx.array_i32([i32(i)], [1])
		nn.save_image('data/predictions/${i}_pred.png', pred.take_axis(sl, 0)) or { panic(err) }
		nn.save_image('data/predictions/${i}_gt.png', val.y.take_axis(sl, 0)) or { panic(err) }
		nn.save_image('data/predictions/${i}_img.png', val.x.take_axis(sl, 0)) or { panic(err) }
	}
	println('sample predictions written to data/predictions/')

	net.save('bsds_hed.safetensors')
	println('model saved to bsds_hed.safetensors')
}
