module nn

import math
import mlx

// metrics.v — evaluation metrics for dense-prediction vision tasks.

// DepthMetrics holds the standard monocular-depth evaluation numbers:
// mean absolute relative error, root mean squared error, and the fraction of
// pixels with max(pred/target, target/pred) < 1.25.
pub struct DepthMetrics {
pub:
	abs_rel  f32
	rmse     f32
	delta125 f32
}

pub fn (m DepthMetrics) str() string {
	return 'AbsRel=${m.abs_rel:.4f}  RMSE=${m.rmse:.4f}  δ<1.25=${m.delta125:.4f}'
}

// depth_metrics computes DepthMetrics between predicted and target depth maps.
pub fn depth_metrics(pred mlx.Array, target mlx.Array) DepthMetrics {
	diff := pred.subtract(target)
	abs_rel := diff.abs().divide(target).mean().item_f32()
	rmse := f32(math.sqrt(f64(diff.square().mean().item_f32())))
	ratio := pred.divide(target).maximum(target.divide(pred))
	delta := mlx.s_lt(ratio, 1.25)
	delta125 := delta.astype(.float32).mean().item_f32()
	return DepthMetrics{
		abs_rel: abs_rel
		rmse: rmse
		delta125: delta125
	}
}

// EdgeMetrics holds edge-detection evaluation numbers: F1 at a fixed 0.5
// threshold and the best F1 over a threshold sweep (a lightweight stand-in
// for ODS, without per-image NMS).
pub struct EdgeMetrics {
pub:
	f1_at_50    f32
	best_f1     f32
	best_thresh f32
}

pub fn (m EdgeMetrics) str() string {
	return 'F1@0.5=${m.f1_at_50:.4f}  bestF1=${m.best_f1:.4f}@${m.best_thresh:.2f}'
}

// edge_metrics computes EdgeMetrics between predicted edge probabilities and
// binary edge targets.
pub fn edge_metrics(pred mlx.Array, target mlx.Array) EdgeMetrics {
	mut best_f1 := f32(0)
	mut best_t := f32(0)
	mut f1_50 := f32(0)
	mut calc := EdgeCalculator{}
	for i in 1 .. 20 {
		t := f32(i) / 20.0
		f1 := calc.f1_at(pred, target, t)
		if t == f32(0.5) {
			f1_50 = f1
		}
		if f1 > best_f1 {
			best_f1 = f1
			best_t = t
		}
	}
	return EdgeMetrics{
		f1_at_50: f1_50
		best_f1: best_f1
		best_thresh: best_t
	}
}

// EdgeCalculator carries the thresholding computation for edge metrics.
pub struct EdgeCalculator {
	eps f32 = 1e-7
}

// f1_at thresholds pred at t and returns the F1 of the positive class.
pub fn (mut c EdgeCalculator) f1_at(pred mlx.Array, target mlx.Array, t f32) f32 {
	hit := mlx.s_gt(pred, t)
	is_pos := mlx.s_gt(target, 0.5)
	tp := hit.logical_and(is_pos).astype(.float32).sum().item_f32()
	fp := hit.logical_and(is_pos.logical_not()).astype(.float32).sum().item_f32()
	fn_ := hit.logical_not().logical_and(is_pos).astype(.float32).sum().item_f32()
	denom := 2 * tp + fp + fn_
	if denom == 0 {
		return 0
	}
	return 2 * tp / denom
}
