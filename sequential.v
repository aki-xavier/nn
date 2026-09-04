module nn

import mlx

// sequential.v — a stack of layers executed in order.

pub struct Sequential {
mut:
	layers []Layer
}

// add appends a layer to the end of the stack.
pub fn (mut net Sequential) add(l Layer) {
	net.layers << l
}

// forward runs the input through every layer in order.
pub fn (mut net Sequential) forward(x mlx.Array) mlx.Array {
	mut out := x
	for mut l in net.layers {
		out = l.forward(out)
	}
	return out
}

// backward propagates the loss gradient through the layers in reverse order.
pub fn (mut net Sequential) backward(grad mlx.Array) {
	mut g := grad
	for i := net.layers.len - 1; i >= 0; i-- {
		g = net.layers[i].backward(g)
	}
}

// predict is forward for inference.
pub fn (mut net Sequential) predict(x mlx.Array) mlx.Array {
	return net.forward(x)
}

// fit trains the network with full-batch gradient descent, printing the loss
// every `log_every` epochs (set 0 to stay quiet).
pub fn (mut net Sequential) fit(x mlx.Array, y mlx.Array, mut criterion Loss, mut opt Optimizer, epochs int, log_every int) {
	net.set_training(true)
	for epoch in 1 .. epochs + 1 {
		pred := net.forward(x)
		l := criterion.loss(pred, y)
		if log_every > 0 && (epoch == 1 || epoch % log_every == 0) {
			println('epoch ${epoch:5d}  loss = ${l.item_f32():.6f}')
		}
		g := criterion.gradient(pred, y)
		net.backward(g)
		opt.step(mut net.layers)
	}
}

// set_training switches all training-sensitive layers (Dropout, BatchNorm)
// between training and inference behaviour.
pub fn (mut net Sequential) set_training(training bool) {
	for mut l in net.layers {
		l.set_training(training)
	}
}

// train_step runs one forward/backward/update cycle and returns the loss.
pub fn (mut net Sequential) train_step(x mlx.Array, y mlx.Array, mut criterion Loss, mut opt Optimizer) f32 {
	pred := net.forward(x)
	l := criterion.loss(pred, y)
	loss_v := l.item_f32()
	g := criterion.gradient(pred, y)
	net.backward(g)
	opt.step(mut net.layers)
	return loss_v
}

// fit_loader trains with mini-batches from a DataLoader, printing the mean
// epoch loss every `log_every` epochs (set 0 to stay quiet).
pub fn (mut net Sequential) fit_loader(mut dl DataLoader, mut criterion Loss, mut opt Optimizer, epochs int, log_every int) {
	net.set_training(true)
	for epoch in 1 .. epochs + 1 {
		dl.reset()
		mut total := f32(0)
		mut batches := 0
		for {
			batch := dl.next() or { break }
			total += net.train_step(batch.x, batch.y, mut criterion, mut opt)
			batches++
		}
		if log_every > 0 && (epoch == 1 || epoch % log_every == 0) {
			println('epoch ${epoch:5d}  loss = ${total / f32(batches):.6f}')
		}
	}
}

// save writes all layer parameters to a safetensors file.
pub fn (mut net Sequential) save(path string) {
	m := mlx.new_map_string_to_array()
	for i, mut l in net.layers {
		l.save_params(m, 'layers.${i}')
	}
	meta := mlx.new_map_string_to_string()
	mlx.save_safetensors(path, m, meta)
	m.free()
	meta.free()
}

// load restores all layer parameters from a safetensors file.  The network
// must already have the same architecture as the saved one.
pub fn (mut net Sequential) load(path string) {
	m, meta := mlx.load_safetensors(path)
	for i, mut l in net.layers {
		l.load_params(m, 'layers.${i}')
	}
	m.free()
	meta.free()
}
