module nn

import mlx

// loss.v — loss functions.  `loss` returns the scalar loss and `gradient`
// returns d(loss)/d(pred), the seed for backpropagation.
//
// Like Layer, Loss is a sum type with match dispatch (see layer.v for why
// interfaces cannot be used together with the mlx import in this V version).
// All receivers are `mut`: value-receiver methods on empty structs hit the
// same V parser bug when combined with a sum type and the mlx import.

pub type Loss = BerHuLoss
	| L1Loss
	| MSELoss
	| ScaleInvariantLoss
	| SoftmaxCrossEntropyLoss
	| WeightedBCELoss

// loss computes the scalar loss between prediction and target.
pub fn (mut l Loss) loss(pred mlx.Array, target mlx.Array) mlx.Array {
	match mut l {
		MSELoss {
			return l.loss(pred, target)
		}
		SoftmaxCrossEntropyLoss {
			return l.loss(pred, target)
		}
		L1Loss {
			return l.loss(pred, target)
		}
		BerHuLoss {
			return l.loss(pred, target)
		}
		ScaleInvariantLoss {
			return l.loss(pred, target)
		}
		WeightedBCELoss {
			return l.loss(pred, target)
		}
	}
}

// gradient computes the gradient of the loss w.r.t. the prediction.
pub fn (mut l Loss) gradient(pred mlx.Array, target mlx.Array) mlx.Array {
	match mut l {
		MSELoss {
			return l.gradient(pred, target)
		}
		SoftmaxCrossEntropyLoss {
			return l.gradient(pred, target)
		}
		L1Loss {
			return l.gradient(pred, target)
		}
		BerHuLoss {
			return l.gradient(pred, target)
		}
		ScaleInvariantLoss {
			return l.gradient(pred, target)
		}
		WeightedBCELoss {
			return l.gradient(pred, target)
		}
	}
}

// MSELoss is the mean squared error over all elements.
pub struct MSELoss {}

pub fn (mut l MSELoss) loss(pred mlx.Array, target mlx.Array) mlx.Array {
	return pred.subtract(target).square().mean()
}

pub fn (mut l MSELoss) gradient(pred mlx.Array, target mlx.Array) mlx.Array {
	n := f64(pred.size())
	return mlx.s_mul(pred.subtract(target), 2.0 / n)
}

// SoftmaxCrossEntropyLoss combines a softmax over the last axis of `pred`
// (raw logits, shape [batch, classes]) with cross entropy against one-hot
// targets of the same shape.
pub struct SoftmaxCrossEntropyLoss {}

pub fn (mut l SoftmaxCrossEntropyLoss) loss(pred mlx.Array, target mlx.Array) mlx.Array {
	batch := f64(pred.shape()[0])
	p := pred.softmax_axis(-1, false)
	return mlx.s_mul(target.multiply(p.log()).sum(), -1.0 / batch)
}

pub fn (mut l SoftmaxCrossEntropyLoss) gradient(pred mlx.Array, target mlx.Array) mlx.Array {
	batch := f64(pred.shape()[0])
	p := pred.softmax_axis(-1, false)
	return mlx.s_mul(p.subtract(target), 1.0 / batch)
}

// L1Loss is the mean absolute error; common for dense regression (depth).
pub struct L1Loss {}

pub fn (mut l L1Loss) loss(pred mlx.Array, target mlx.Array) mlx.Array {
	return pred.subtract(target).abs().mean()
}

pub fn (mut l L1Loss) gradient(pred mlx.Array, target mlx.Array) mlx.Array {
	n := f64(pred.size())
	sign := mlx.where(mlx.s_gt(pred.subtract(target), 0.0), mlx.ones_like(pred), mlx.s_mul(mlx.ones_like(pred), -1.0))
	return mlx.s_mul(sign, 1.0 / n)
}

// BerHuLoss (reverse Huber) is quadratic for |e| <= c and linear beyond it,
// a standard depth-estimation loss.  c defaults to 0.2·max|e| per batch
// (c = 0 pins it to a fixed threshold).
pub struct BerHuLoss {
pub:
	c f32
}

pub fn (mut l BerHuLoss) loss(pred mlx.Array, target mlx.Array) mlx.Array {
	e := pred.subtract(target).abs()
	c := l.threshold(e)
	small := mlx.s_mul(e.square(), 0.5 / c)
	big := mlx.s_rsub(mlx.s_mul(e, 1.0), c / 2.0)
	return mlx.where(mlx.s_le(e, c), small, big).mean()
}

pub fn (mut l BerHuLoss) gradient(pred mlx.Array, target mlx.Array) mlx.Array {
	diff := pred.subtract(target)
	e := diff.abs()
	c := l.threshold(e)
	n := f64(pred.size())
	small := mlx.s_mul(diff, 1.0 / c)
	big := mlx.where(mlx.s_gt(diff, 0.0), mlx.ones_like(diff), mlx.s_mul(mlx.ones_like(diff), -1.0))
	return mlx.s_mul(mlx.where(mlx.s_le(e, c), small, big), 1.0 / n)
}

// threshold returns c: the configured value, or 0.2·max|e| when c == 0.
pub fn (l BerHuLoss) threshold(e mlx.Array) f64 {
	if l.c > 0 {
		return f64(l.c)
	}
	return f64(e.max().item_f32()) * 0.2
}

// ScaleInvariantLoss is Eigen et al.'s scale-invariant depth loss:
// mean(e²) - λ/n²·(Σe)² with e = pred - target (usually in log space).
pub struct ScaleInvariantLoss {
pub:
	lambda f32 = 0.5
}

pub fn (mut l ScaleInvariantLoss) loss(pred mlx.Array, target mlx.Array) mlx.Array {
	n := f64(pred.size())
	e := pred.subtract(target)
	mse := e.square().mean()
	scale := mlx.s_mul(e.sum().square(), f64(l.lambda) / (n * n))
	return mse.subtract(scale)
}

pub fn (mut l ScaleInvariantLoss) gradient(pred mlx.Array, target mlx.Array) mlx.Array {
	n := f64(pred.size())
	e := pred.subtract(target)
	sum_e := e.sum().item_f32()
	return mlx.s_mul(e.subtract(mlx.full_value(pred.shape(), f32(f64(l.lambda) * f64(sum_e) / n), .float32)), 2.0 / n)
}

// WeightedBCELoss is binary cross entropy over probabilities in (0, 1) with
// a positive-class weight — the HED-style edge loss, where edge pixels are
// rare and get weight w_pos = (1 - edge_fraction) / edge_fraction.
pub struct WeightedBCELoss {
pub:
	w_pos f32 = 1.0
	eps   f32 = 1e-7
}

pub fn (mut l WeightedBCELoss) loss(pred mlx.Array, target mlx.Array) mlx.Array {
	p := mlx.s_clip(pred, f64(l.eps), 1.0 - f64(l.eps))
	pos := mlx.s_mul(target.multiply(p.log()), -f64(l.w_pos))
	neg := mlx.s_rsub(target, 1.0).multiply(mlx.s_rsub(p, 1.0).log())
	return pos.subtract(neg).mean()
}

pub fn (mut l WeightedBCELoss) gradient(pred mlx.Array, target mlx.Array) mlx.Array {
	p := mlx.s_clip(pred, f64(l.eps), 1.0 - f64(l.eps))
	n := f64(pred.size())
	pos := mlx.s_mul(target.divide(p), -f64(l.w_pos))
	neg := mlx.s_rsub(target, 1.0).divide(mlx.s_rsub(p, 1.0))
	return mlx.s_mul(pos.add(neg), 1.0 / n)
}
