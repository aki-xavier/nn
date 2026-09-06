// non-OOP — the vjp trampolines must be top-level fns: MLX autograd takes a
// plain C function pointer (mlx.Func) that cannot capture the layer instance.
module nn

import math
import mlx

// sequence.v — sequence layers: multi-head self-attention and LSTM.
//
// Both are single differentiable blocks implemented through vjp trampolines
// (the unrolled loops live inside the traced function), with parameters
// exposed through the standard Layer protocol so optimizers and persistence
// work unchanged.  Inputs are [n, t, c]; weights are stored [out, in] so
// the forward is x·wᵀ.

// ============================================================================
// Attention: multi-head self-attention.  x [n, t, c] -> [n, t, c].
// ============================================================================

pub struct Attention {
pub:
	dim    int
	heads  int
	causal bool
mut:
	wq  mlx.Array // [c, c] (out, in)
	wk  mlx.Array
	wv  mlx.Array
	wo  mlx.Array
	bq  mlx.Array // [c]
	bk  mlx.Array
	bv  mlx.Array
	bo  mlx.Array
	x   mlx.Array
	dwq mlx.Array
	dwk mlx.Array
	dwv mlx.Array
	dwo mlx.Array
	dbq mlx.Array
	dbk mlx.Array
	dbv mlx.Array
	dbo mlx.Array
}

pub fn new_attention(dim int, heads int, causal bool, seed u64) Attention {
	key := mlx.random_key(seed)
	defer {
		key.free()
	}
	scale := f32(1.0 / math.sqrt(f64(dim)))
	return Attention{
		dim: dim
		heads: heads
		causal: causal
		wq: mlx.random_normal([dim, dim], .float32, 0.0, scale, key)
		wk: mlx.random_normal([dim, dim], .float32, 0.0, scale, key)
		wv: mlx.random_normal([dim, dim], .float32, 0.0, scale, key)
		wo: mlx.random_normal([dim, dim], .float32, 0.0, scale, key)
		bq: mlx.zeros([dim], .float32)
		bk: mlx.zeros([dim], .float32)
		bv: mlx.zeros([dim], .float32)
		bo: mlx.zeros([dim], .float32)
	}
}

// attention_fwd is the autograd trampoline; xs = [x, wq..bo, cfg], cfg int32
// [heads, causal, 0].
fn attention_fwd(xs []mlx.Array) []mlx.Array {
	cfg := xs[9].data_i32()
	return [
		multihead_attention(xs[0], cfg[0], cfg[1] == 1, xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7], xs[8]),
	]
}

// multihead_attention computes softmax(q·kᵀ/√d)·v with an optional causal
// mask; weights are [out, in] and biases [out], head dim = c/heads.
fn multihead_attention(x mlx.Array, heads int, causal bool, wq mlx.Array, wk mlx.Array, wv mlx.Array, wo mlx.Array, bq mlx.Array, bk mlx.Array, bv mlx.Array, bo mlx.Array) mlx.Array {
	shape := x.shape()
	n := shape[0]
	t := shape[1]
	c := shape[2]
	hd := c / heads
	mut q := mlx.einsum('nti,oi->nto', [x, wq]).add(bq)
	mut k := mlx.einsum('nti,oi->nto', [x, wk]).add(bk)
	mut v := mlx.einsum('nti,oi->nto', [x, wv]).add(bv)
	q = q.reshape([n, t, heads, hd]).transpose_axes([0, 2, 1, 3])
	k = k.reshape([n, t, heads, hd]).transpose_axes([0, 2, 1, 3])
	v = v.reshape([n, t, heads, hd]).transpose_axes([0, 2, 1, 3])
	mut scores := mlx.einsum('nhkd,nhld->nhkl', [q, k]).multiply(mlx.f32_scalar(f32(1.0 / math.sqrt(f64(hd)))))
	if causal {
		scores = scores.add(causal_mask(t))
	}
	p := scores.softmax_axis(-1, false)
	ov := mlx.einsum('nhkl,nhld->nhkd', [p, v])
	oc := ov.transpose_axes([0, 2, 1, 3]).reshape([n, t, c])
	return mlx.einsum('nti,oi->nto', [oc, wo]).add(bo)
}

// causal_mask returns a [t, t] add-mask (0 allowed, -1e9 forbidden).
fn causal_mask(t int) mlx.Array {
	mut m := []f32{len: t * t}
	for i in 0 .. t {
		for j in 0 .. t {
			if j > i {
				m[i * t + j] = -1e9
			}
		}
	}
	return mlx.array_f32(m, [t, t])
}

pub fn (mut l Attention) forward(x mlx.Array) mlx.Array {
	l.x = x
	return multihead_attention(x, l.heads, l.causal, l.wq, l.wk, l.wv, l.wo, l.bq, l.bk, l.bv, l.bo)
}

pub fn (mut l Attention) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(l.heads), i32(b2i(l.causal)), i32(0)], [3])
	_, vjps := mlx.vjp(attention_fwd, [l.x, l.wq, l.wk, l.wv, l.wo, l.bq, l.bk, l.bv, l.bo, cfg], [
		grad,
	])
	l.dwq = vjps[1]
	l.dwk = vjps[2]
	l.dwv = vjps[3]
	l.dwo = vjps[4]
	l.dbq = vjps[5]
	l.dbk = vjps[6]
	l.dbv = vjps[7]
	l.dbo = vjps[8]
	return vjps[0]
}

pub fn (mut l Attention) params() []mlx.Array {
	return [l.wq, l.wk, l.wv, l.wo, l.bq, l.bk, l.bv, l.bo]
}

pub fn (mut l Attention) grads() []mlx.Array {
	return [l.dwq, l.dwk, l.dwv, l.dwo, l.dbq, l.dbk, l.dbv, l.dbo]
}

pub fn (mut l Attention) set_params(ps []mlx.Array) {
	l.wq = ps[0]
	l.wk = ps[1]
	l.wv = ps[2]
	l.wo = ps[3]
	l.bq = ps[4]
	l.bk = ps[5]
	l.bv = ps[6]
	l.bo = ps[7]
}

pub fn (mut l Attention) set_training(training bool) {}

pub fn (mut l Attention) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.wq', l.wq)
	m.insert('${prefix}.wk', l.wk)
	m.insert('${prefix}.wv', l.wv)
	m.insert('${prefix}.wo', l.wo)
	m.insert('${prefix}.bq', l.bq)
	m.insert('${prefix}.bk', l.bk)
	m.insert('${prefix}.bv', l.bv)
	m.insert('${prefix}.bo', l.bo)
}

pub fn (mut l Attention) load_params(m mlx.MapStringToArray, prefix string) {
	l.wq = reshape_to(m.get('${prefix}.wq'), [l.dim, l.dim], '${prefix}.wq')
	l.wk = reshape_to(m.get('${prefix}.wk'), [l.dim, l.dim], '${prefix}.wk')
	l.wv = reshape_to(m.get('${prefix}.wv'), [l.dim, l.dim], '${prefix}.wv')
	l.wo = reshape_to(m.get('${prefix}.wo'), [l.dim, l.dim], '${prefix}.wo')
	l.bq = reshape_to(m.get('${prefix}.bq'), [l.dim], '${prefix}.bq')
	l.bk = reshape_to(m.get('${prefix}.bk'), [l.dim], '${prefix}.bk')
	l.bv = reshape_to(m.get('${prefix}.bv'), [l.dim], '${prefix}.bv')
	l.bo = reshape_to(m.get('${prefix}.bo'), [l.dim], '${prefix}.bo')
	for a in [l.wq, l.wk, l.wv, l.wo, l.bq, l.bk, l.bv, l.bo] {
		a.eval()
	}
}

// b2i converts a bool config flag to int.
fn b2i(v bool) int {
	if v {
		return 1
	}
	return 0
}

// ============================================================================
// LSTM: x [n, t, c] -> hidden states [n, t, h].  Gate order:
// input, forget, candidate, output.
// ============================================================================

pub struct LSTM {
pub:
	input_size  int
	hidden_size int
mut:
	w_ih  mlx.Array // [4h, c]
	w_hh  mlx.Array // [4h, h]
	b     mlx.Array // [4h]
	x     mlx.Array
	dw_ih mlx.Array
	dw_hh mlx.Array
	db    mlx.Array
}

pub fn new_lstm(input_size int, hidden_size int, seed u64) LSTM {
	key := mlx.random_key(seed)
	defer {
		key.free()
	}
	scale := f32(1.0 / math.sqrt(f64(hidden_size)))
	return LSTM{
		input_size: input_size
		hidden_size: hidden_size
		w_ih: mlx.random_uniform(mlx.f32_scalar(-scale), mlx.f32_scalar(scale), [
			4 * hidden_size,
			input_size,
		], .float32, key)
		w_hh: mlx.random_uniform(mlx.f32_scalar(-scale), mlx.f32_scalar(scale), [
			4 * hidden_size,
			hidden_size,
		], .float32, key)
		b: mlx.zeros([4 * hidden_size], .float32)
	}
}

fn gate_index(hidden int, g int) mlx.Array {
	lo := g * hidden
	return mlx.array_i32([]int{len: hidden, init: lo + index}.map(i32(it)), [hidden])
}

// lstm_fwd is the autograd trampoline; xs = [x, w_ih, w_hh, b, cfg] with cfg
// int32 [hidden].
fn lstm_fwd(xs []mlx.Array) []mlx.Array {
	return [lstm_trace(xs[0], xs[4].data_i32()[0], xs[1], xs[2], xs[3])]
}

// lstm_trace runs the unrolled LSTM recurrence.
fn lstm_trace(x mlx.Array, hidden int, w_ih mlx.Array, w_hh mlx.Array, b mlx.Array) mlx.Array {
	shape := x.shape()
	n := shape[0]
	t := shape[1]
	c := shape[2]
	mut h := mlx.zeros([n, hidden], .float32)
	mut c_s := mlx.zeros([n, hidden], .float32)
	mut outs := []mlx.Array{}
	ih_t := w_ih.transpose()
	hh_t := w_hh.transpose()
	for i in 0 .. t {
		xt := x.take_axis(mlx.array_i32([i32(i)], [1]), 1).reshape([n, c])
		gates := xt.matmul(ih_t).add(h.matmul(hh_t)).add(b)
		gi := gates.take_axis(gate_index(hidden, 0), 1).sigmoid()
		gf := gates.take_axis(gate_index(hidden, 1), 1).sigmoid()
		gc := gates.take_axis(gate_index(hidden, 2), 1).tanh()
		gate_o := gates.take_axis(gate_index(hidden, 3), 1).sigmoid()
		c_s = gf.multiply(c_s).add(gi.multiply(gc))
		h = gate_o.multiply(c_s.tanh())
		outs << h.reshape([n, 1, hidden])
	}
	return mlx.concatenate(outs, 1)
}

pub fn (mut l LSTM) forward(x mlx.Array) mlx.Array {
	l.x = x
	return lstm_trace(x, l.hidden_size, l.w_ih, l.w_hh, l.b)
}

pub fn (mut l LSTM) backward(grad mlx.Array) mlx.Array {
	cfg := mlx.array_i32([i32(l.hidden_size)], [1])
	_, vjps := mlx.vjp(lstm_fwd, [l.x, l.w_ih, l.w_hh, l.b, cfg], [grad])
	l.dw_ih = vjps[1]
	l.dw_hh = vjps[2]
	l.db = vjps[3]
	return vjps[0]
}

pub fn (mut l LSTM) params() []mlx.Array {
	return [l.w_ih, l.w_hh, l.b]
}

pub fn (mut l LSTM) grads() []mlx.Array {
	return [l.dw_ih, l.dw_hh, l.db]
}

pub fn (mut l LSTM) set_params(ps []mlx.Array) {
	l.w_ih = ps[0]
	l.w_hh = ps[1]
	l.b = ps[2]
}

pub fn (mut l LSTM) set_training(training bool) {}

pub fn (mut l LSTM) save_params(m mlx.MapStringToArray, prefix string) {
	m.insert('${prefix}.w_ih', l.w_ih)
	m.insert('${prefix}.w_hh', l.w_hh)
	m.insert('${prefix}.b', l.b)
}

pub fn (mut l LSTM) load_params(m mlx.MapStringToArray, prefix string) {
	l.w_ih = reshape_to(m.get('${prefix}.w_ih'), [4 * l.hidden_size, l.input_size], '${prefix}.w_ih')
	l.w_hh = reshape_to(m.get('${prefix}.w_hh'), [4 * l.hidden_size, l.hidden_size], '${prefix}.w_hh')
	l.b = reshape_to(m.get('${prefix}.b'), [4 * l.hidden_size], '${prefix}.b')
	l.w_ih.eval()
	l.w_hh.eval()
	l.b.eval()
}
