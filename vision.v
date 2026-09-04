module nn

import mlx
import stbi

// vision.v — image decoding helpers for vision datasets (PNG/JPEG via stb).

// ImageData is a decoded image as an NHWC float32 tensor scaled to [0, 1].
pub struct ImageData {
pub:
	tensor   mlx.Array // [1, h, w, channels]
	width    int
	height   int
	channels int
}

// load_image decodes an image file into an ImageData.  `channels` is the
// desired channel count (1 = grayscale, 3 = RGB); 0 keeps the file's own.
pub fn load_image(path string, channels int) !ImageData {
	img := stbi.load(path, desired_channels: channels) or { return error('nn: ${err}') }
	defer {
		img.free()
	}
	n := img.width * img.height * img.nr_channels
	mut px := []f32{len: n}
	unsafe {
		for i in 0 .. n {
			px[i] = f32(img.data[i]) / 255.0
		}
	}
	return ImageData{
		tensor: mlx.array_f32(px, [1, img.height, img.width, img.nr_channels])
		width: img.width
		height: img.height
		channels: img.nr_channels
	}
}

// stack_images batches decoded images into [n, h, w, channels]; all images
// must share the same shape.
pub fn stack_images(images []ImageData) mlx.Array {
	tensors := images.map(it.tensor)
	return mlx.concatenate(tensors, 0)
}

// flip_horizontal mirrors an NHWC tensor along the width axis.
pub fn flip_horizontal(x mlx.Array) mlx.Array {
	shape := x.shape()
	w := shape[2]
	idx := mlx.array_i32([]int{len: w, init: w - 1 - index}.map(i32(it)), [w])
	return x.take_axis(idx, 2)
}

// save_image writes an NHWC tensor [1, h, w, c] with values in [0, 1] to a
// PNG file (c=1 grayscale, c=3 RGB).
pub fn save_image(path string, t mlx.Array) ! {
	shape := t.shape()
	if shape.len != 4 || shape[0] != 1 {
		return error('nn: save_image expects [1, h, w, c], got ${shape}')
	}
	c := shape[3]
	if c != 1 && c != 3 {
		return error('nn: save_image supports 1 or 3 channels, got ${c}')
	}
	data := t.data_f32()
	mut px := []u8{len: data.len}
	for i, v in data {
		mut u := v * 255.0
		if u < 0 {
			u = 0
		}
		if u > 255 {
			u = 255
		}
		px[i] = u8(u)
	}
	stbi.stbi_write_png(path, shape[2], shape[1], c, px.data, shape[2] * c) or {
		return error('nn: stbi write failed: ${err}')
	}
}

// resize_nearest resizes an NHWC tensor to (height, width) by nearest
// neighbour, using axis gathers.
pub fn resize_nearest(x mlx.Array, height int, width int) mlx.Array {
	shape := x.shape()
	h_in := shape[1]
	w_in := shape[2]
	row_idx := mlx.array_i32([]int{len: height, init: index * h_in / height}.map(i32(it)), [
		height,
	])
	col_idx := mlx.array_i32([]int{len: width, init: index * w_in / width}.map(i32(it)), [
		width,
	])
	return x.take_axis(row_idx, 1).take_axis(col_idx, 2)
}
