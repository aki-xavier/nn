module nn

import mlx

// sequential.v — a stack of layers executed in order.

pub struct Sequential {
pub mut:
	last_grad_norm f32 // L2 global grad norm recorded by the last train_step
mut:
	layers    []Layer
	scheduler LRScheduler = StepLR{ lr: 0, step_size: 1 }
	has_sched bool
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
// `last_grad_norm` records the pre-clip global gradient norm.
pub fn (mut net Sequential) train_step(x mlx.Array, y mlx.Array, mut criterion Loss, mut opt Optimizer) f32 {
	pred := net.forward(x)
	l := criterion.loss(pred, y)
	loss_v := l.item_f32()
	g := criterion.gradient(pred, y)
	net.backward(g)
	net.last_grad_norm = opt.grad_norm(mut net.layers)
	opt.step(mut net.layers)
	return loss_v
}

// use_scheduler attaches an LR scheduler, applied at the start of every epoch.
pub fn (mut net Sequential) use_scheduler(mut sched LRScheduler) {
	net.scheduler = sched
	net.has_sched = true
}

// fit_loader trains with mini-batches from a DataLoader, printing the mean
// epoch loss every `log_every` epochs (set 0 to stay quiet).
pub fn (mut net Sequential) fit_loader(mut dl DataLoader, mut criterion Loss, mut opt Optimizer, epochs int, log_every int) {
	net.set_training(true)
	for epoch in 1 .. epochs + 1 {
		if net.has_sched {
			opt.schedule(epoch - 1, mut net.scheduler)
		}
		dl.reset()
		mut total := f32(0)
		mut batches := 0
		for {
			batch := dl.next() or { break }
			total += net.train_step(batch.x, batch.y, mut criterion, mut opt)
			batches++
		}
		if log_every > 0 && (epoch == 1 || epoch % log_every == 0) {
			println('epoch ${epoch:5d}  loss = ${total / f32(batches):.6f}  |grad| = ${net.last_grad_norm:.4f}')
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
	net.load_map(m)
	m.free()
	meta.free()
}

// load_map restores all layer parameters from an open tensor map using the
// internal key convention (`layers.{i}.w` / `.b` / ...).
pub fn (mut net Sequential) load_map(m mlx.MapStringToArray) {
	for i, mut l in net.layers {
		l.load_params(m, 'layers.${i}')
	}
}

// load_checkpoint loads parameters from an external checkpoint according to
// `rules` (name mapping + optional layout permutation), then validates the
// result.  Unknown checkpoint names panic with the list of available keys;
// shape mismatches panic with expected/got shapes.
pub fn (mut net Sequential) load_checkpoint(ckpt Checkpoint, rules []LoadRule) {
	mapped := mlx.new_map_string_to_array()
	defer {
		mapped.free()
	}
	for rule in rules {
		mapped.insert(rule.to, ckpt.tensor(rule.from, rule.perm))
	}
	net.load_map(mapped)
}

// to_dtype converts all trainable parameters to `dt` (e.g. half-precision
// weights for inference).  Non-trainable state (BatchNorm running stats)
// stays in float32.
pub fn (mut net Sequential) to_dtype(dt mlx.Dtype) {
	for mut l in net.layers {
		ps := l.params()
		if ps.len == 0 {
			continue
		}
		mut np := []mlx.Array{cap: ps.len}
		for p in ps {
			np << p.astype(dt)
		}
		l.set_params(np)
	}
}

// half converts trainable parameters to float16.
pub fn (mut net Sequential) half() {
	net.to_dtype(.float16)
}

// to_float32 converts trainable parameters back to float32.
pub fn (mut net Sequential) to_float32() {
	net.to_dtype(.float32)
}

// param_count returns the total number of trainable parameters.
pub fn (mut net Sequential) param_count() int {
	mut total := 0
	for mut l in net.layers {
		for p in l.params() {
			total += int(p.size())
		}
	}
	return total
}

// forward_taps runs the network and returns the outputs right after each
// layer index in `taps` — side outputs for HED-style edge detection and
// FPN-style multi-scale features.
pub fn (mut net Sequential) forward_taps(x mlx.Array, taps []int) []mlx.Array {
	mut out := x
	mut res := []mlx.Array{}
	for i, mut l in net.layers {
		out = l.forward(out)
		if i in taps {
			res << out
		}
	}
	return res
}
