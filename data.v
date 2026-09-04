module nn

import mlx
import rand

// data.v — mini-batch data pipeline.
//
// A Dataset holds the full design matrices (first axis = samples); a
// DataLoader yields shuffled mini-batches via take_axis gathers, so batches
// stay on-device without host-side slicing.

pub struct Dataset {
pub:
	x mlx.Array // [n, ...]
	y mlx.Array // [n, ...]
}

// DataLoader iterates a Dataset in mini-batches.
pub struct DataLoader {
pub:
	dataset    Dataset
	batch_size int
	shuffle    bool = true
	drop_last  bool
mut:
	order []int
	pos   int
}

pub fn new_dataloader(dataset Dataset, batch_size int, shuffle bool) DataLoader {
	return DataLoader{
		dataset: dataset
		batch_size: batch_size
		shuffle: shuffle
		order: []int{len: int(dataset.x.shape()[0]), init: index}
	}
}

// reset starts a new epoch, reshuffling when enabled.
pub fn (mut dl DataLoader) reset() {
	dl.pos = 0
	if dl.shuffle {
		rand.shuffle(mut dl.order) or {}
	}
}

// next returns the next batch, or none when the epoch is exhausted.
pub fn (mut dl DataLoader) next() ?Batch {
	n := dl.order.len
	if dl.pos >= n {
		return none
	}
	mut end := dl.pos + dl.batch_size
	if end > n {
		if dl.drop_last {
			dl.pos = n
			return none
		}
		end = n
	}
	idx := mlx.array_i32(dl.order[dl.pos..end].map(i32(it)), [end - dl.pos])
	dl.pos = end
	return Batch{
		x: dl.dataset.x.take_axis(idx, 0)
		y: dl.dataset.y.take_axis(idx, 0)
	}
}

// Batch is one mini-batch pair.
pub struct Batch {
pub:
	x mlx.Array
	y mlx.Array
}
