module main

import mlx
import nn

// Train a tiny MLP on XOR, persist the weights, then reload them into a
// fresh network and run inference.

fn main() {
	println('mlx ${mlx.version()}  gpu: ${mlx.gpu_available()}')

	// XOR dataset: 4 samples, 2 inputs, 1 output.
	x := mlx.array_f32([f32(0), 0, 0, 1, 1, 0, 1, 1], [4, 2])
	y := mlx.array_f32([f32(0), 1, 1, 0], [4, 1])

	// 2 -> 8 (tanh) -> 1 (sigmoid)
	mut net := nn.Sequential{}
	net.add(nn.new_linear(2, 8, 42))
	net.add(nn.Tanh{})
	net.add(nn.new_linear(8, 1, 43))
	net.add(nn.Sigmoid{})

	println('before training:')
	println(net.predict(x))

	mut opt := nn.Optimizer(nn.SGD{
		lr: 0.5
	})
	mut mse := nn.MSELoss{}
	net.fit(x, y, mut mse, mut opt, 5000, 500)

	println('after training:')
	println(net.predict(x))

	// Persist and reload the trained weights.
	path := 'xor.safetensors'
	net.save(path)
	println('saved weights to ${path}')

	mut net2 := nn.Sequential{}
	net2.add(nn.new_linear(2, 8, 0))
	net2.add(nn.Tanh{})
	net2.add(nn.new_linear(8, 1, 0))
	net2.add(nn.Sigmoid{})
	net2.load(path)

	println('predictions from reloaded network (expect ~[0, 1, 1, 0]):')
	println(net2.predict(x))
}
