module nn

// backbones.v — architecture presets.  These are ordinary builders that
// return ready-to-train Sequential/Module nets; each block mirrors the
// standard layout of the well-known backbone (VGG-style stacks, ResNet-style
// blocks, two-level U-Net) at channel counts scaled for a small framework.
// Weights are seeded for reproducibility and can be swapped for pretrained
// checkpoints via Sequential.load_checkpoint.

// vgg16_block stacks `convs` conv3x3 -> relu pairs; the first conv maps
// `in_ch` to `channels`, the rest stay at `channels`.
fn vgg16_block(in_ch int, channels int, convs int, seed u64) []Layer {
	mut out := []Layer{}
	for i in 0 .. convs {
		ci := if i == 0 { in_ch } else { channels }
		out << Layer(new_conv2d(ci, channels, 3, 1, 1, seed + u64(i)))
		out << Layer(ReLU{})
	}
	return out
}

// vgg16_lite builds a VGG-ish classifier backbone:
// conv 3->64 x2, pool, 64->128 x2, pool, 128->256 x3, pool, 256->512 x3,
// pool, 512->512 x3, pool, flatten, linear(512 -> classes).
pub fn vgg16_lite(classes int, seed u64) Sequential {
	mut net := Sequential{}
	net.add(new_conv2d(3, 64, 3, 1, 1, seed))
	net.add(ReLU{})
	net.add(new_conv2d(64, 64, 3, 1, 1, seed + 1))
	net.add(ReLU{})
	net.add(new_max_pool2d(2))
	for l in vgg16_block(64, 128, 2, seed + 10) {
		net.add(l)
	}
	net.add(new_max_pool2d(2))
	for l in vgg16_block(128, 256, 3, seed + 20) {
		net.add(l)
	}
	net.add(new_max_pool2d(2))
	for l in vgg16_block(256, 512, 3, seed + 30) {
		net.add(l)
	}
	net.add(new_max_pool2d(2))
	for l in vgg16_block(512, 512, 3, seed + 40) {
		net.add(l)
	}
	net.add(new_max_pool2d(2))
	net.add(GlobalAvgPool2d{})
	net.add(Flatten{})
	net.add(new_linear(512, classes, seed + 99))
	return net
}

// resnet18_lite builds a ResNet-18-style image classifier:
// stem conv7x7 s2 (3->32), pool, two BasicBlocks per stage at
// [32, 64, 128, 256] with stride-2 entry, global pool, linear head.
pub fn resnet18_lite(classes int, seed u64) Sequential {
	mut net := Sequential{}
	net.add(new_conv2d(3, 32, 7, 2, 3, seed))
	net.add(new_group_norm(32, 1))
	net.add(ReLU{})
	net.add(new_max_pool2d(2))
	stages := [64, 128, 256]
	mut in_ch := 32
	for s in stages {
		// stage entry: stride-2 downsample conv (outside the residual so the
		// identity shortcut stays shape-compatible)
		net.add(new_conv2d(in_ch, s, 3, 2, 1, seed + u64(s)))
		net.add(new_group_norm(s, 1))
		net.add(ReLU{})
		// two identity basic blocks
		for j in 0 .. 2 {
			mut blk := Module{}
			blk.add(new_conv2d(s, s, 3, 1, 1, seed + u64(s) + u64(j) * 20 + 13))
			blk.add(new_group_norm(s, 1))
			blk.add(ReLU{})
			blk.add(new_conv2d(s, s, 3, 1, 1, seed + u64(s) + u64(j) * 20 + 19))
			blk.add(new_group_norm(s, 1))
			net.add(Layer(new_residual([Layer(blk)])))
		}
		in_ch = s
	}
	net.add(GlobalAvgPool2d{})
	net.add(Flatten{})
	net.add(new_linear(256, classes, seed + 99))
	return net
}

// hed_unet builds the two-level U-Net used by the BSDS500 edge example
// (Skip-container nesting); it takes a single-channel image stack and
// returns edge likelihoods.
pub fn hed_unet(seed u64) Sequential {
	mut net := Sequential{}
	net.add(new_conv2d(1, 32, 3, 1, 1, seed))
	net.add(ReLU{})
	net.add(new_skip([
		Layer(new_max_pool2d(2)),
		Layer(new_conv2d(32, 64, 3, 1, 1, seed + 1)),
		Layer(ReLU{}),
		Layer(new_skip([
			Layer(new_max_pool2d(2)),
			Layer(new_conv2d(64, 128, 3, 1, 1, seed + 2)),
			Layer(ReLU{}),
			Layer(new_upsample2d(2)),
			Layer(new_conv2d(128, 64, 3, 1, 1, seed + 3)),
		])),
		Layer(ReLU{}),
		Layer(new_upsample2d(2)),
		Layer(new_conv2d(128, 32, 3, 1, 1, seed + 4)),
	]))
	net.add(ReLU{})
	net.add(new_conv2d(64, 32, 3, 1, 1, seed + 5))
	net.add(ReLU{})
	net.add(new_conv2d(32, 1, 1, 1, 0, seed + 6))
	net.add(Sigmoid{})
	return net
}
