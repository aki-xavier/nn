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
