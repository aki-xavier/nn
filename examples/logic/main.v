module main

import mlx
import nn

// Differentiable logic gate network demo: train boolean functions with
// relaxed (soft) gates, then read out the learned circuit (argmax gates) and
// evaluate it as exact boolean logic.

struct LogicDemo {
mut:
	width  int
	epochs int
}

fn (d LogicDemo) bool_table(n_in int) mlx.Array {
	n := 1 << n_in
	mut xs := []f32{}
	for i in 0 .. n {
		for b in 0 .. n_in {
			xs << f32((i >> b) & 1)
		}
	}
	return mlx.array_f32(xs, [n, n_in])
}

fn (d LogicDemo) fmtv(v []f32) string {
	mut s := '['
	for x in v {
		s += '${x:.3f} '
	}
	return s + ']'
}

fn (d LogicDemo) build_net(n_in int, seed u64) nn.Sequential {
	mut net := nn.Sequential{}
	net.add(nn.new_logic_gate_layer_covered(n_in, d.width, true, seed))
	net.add(nn.new_logic_gate_layer_covered(n_in + d.width, d.width, true, seed + 1))
	net.add(nn.new_logic_gate_layer_covered(n_in + 2 * d.width, 1, false, seed + 2))
	return net
}

// discretized evaluates the net with argmax gates (exact boolean).
fn (d LogicDemo) discretized(mut net nn.Sequential, x mlx.Array) []f32 {
	return nn.logic_discretized_forward(mut net, x).data_f32()
}

fn (d LogicDemo) gate_names(net nn.Sequential, layer int) []string {
	return nn.logic_gate_names(net, layer)
}

fn (d LogicDemo) train_case(name string, n_in int, targets []f32) nn.Sequential {
	x := d.bool_table(n_in)
	n := 1 << n_in
	y := mlx.array_f32(targets, [n, 1])
	mut net := d.build_net(n_in, 41)
	mut dl := nn.new_dataloader(nn.Dataset{
		x: x
		y: y
	}, n, true)
	mut mse := nn.Loss(nn.MSELoss{})
	mut opt := nn.Optimizer(nn.Adam{
		lr: 0.05
	})
	net.fit_loader(mut dl, mut mse, mut opt, d.epochs, 0)
	net.set_training(false)
	soft := net.predict(x).data_f32()
	mut err := f32(0)
	for i in 0 .. n {
		mut diff := soft[i] - targets[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > err {
			err = diff
		}
	}
	hard := d.discretized(mut net, x)
	println('${name}: soft max err ${err:.4f}  hard ${d.fmtv(hard)}  target ${d.fmtv(targets)}')
	return net
}

fn (d LogicDemo) run() {
	println('mlx ${mlx.version()}  gpu: ${mlx.gpu_available()}')
	mut xor_net := d.train_case('XOR', 2, [f32(0), 1, 1, 0])
	d.train_case('AND', 2, [f32(0), 0, 0, 1])

	mut maj := []f32{}
	for i in 0 .. 8 {
		a := (i >> 0) & 1
		b := (i >> 1) & 1
		c := (i >> 2) & 1
		maj << f32(if a + b + c >= 2 { 1 } else { 0 })
	}
	d.train_case('MAJ3', 3, maj)

	println('learned XOR circuit -- hidden gates: ${d.gate_names(xor_net, 0)}')
	println('learned XOR circuit -- output gate:  ${d.gate_names(xor_net, 2)}')

	xor_net.save('logic_xor.safetensors')
	mut net2 := d.build_net(2, 0)
	net2.load('logic_xor.safetensors')
	net2.set_training(false)
	before := d.discretized(mut xor_net, d.bool_table(2))
	after := d.discretized(mut net2, d.bool_table(2))
	mut same := true
	for i in 0 .. 4 {
		mut diff := before[i] - after[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			same = false
		}
	}
	if same {
		println('PASS: logic circuit survives save/load')
	} else {
		println('FAIL: weights diverge')
	}
}

fn main() {
	demo := LogicDemo{
		width: 8
		epochs: 800
	}
	demo.run()
}
