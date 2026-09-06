module nn

import math
import mlx

// gradstats.v — global gradient norm accounting and clipping.
//
// Clipping is integrated into the optimizers: SGD/Adam carry a `clip_norm`
// field (0 = disabled) and scale the effective gradient by
// min(1, clip_norm / global_norm) inside the update.  Sequential.train_step
// records the pre-clip norm in `last_grad_norm` for diagnostics.

// grad_norm is the Optimizer-side accessor used by the training loops.
pub fn (mut o Optimizer) grad_norm(mut layers []Layer) f32 {
	match mut o {
		SGD {
			return global_norm(mut layers)
		}
		Adam {
			return global_norm(mut layers)
		}
	}
}

// global_norm returns the L2 norm over every layer gradient.
pub fn global_norm(mut layers []Layer) f32 {
	mut total := f32(0)
	for mut l in layers {
		for g in l.grads() {
			total += g.square().sum().item_f32()
		}
	}
	return f32(math.sqrt(f64(total)))
}

// clip_factor returns the multiplier applying max-norm clipping.
pub fn clip_factor(norm f32, max_norm f32) f32 {
	if max_norm <= 0 || norm <= max_norm {
		return 1.0
	}
	return max_norm / norm
}
