// cga_engine.v — generic Clifford Cl(p,q) algebra engine: blade products
// computed from metric signatures via the canonical form (ascending basis
// index order), powering the conformal algebra Cl(4,1) = CGA.
module nn

import mlx

// bit_count returns the number of set bits.
fn bit_count(mask int) int {
	mut m := mask
	mut n := 0
	for m != 0 {
		m &= m - 1
		n++
	}
	return n
}

// lowest_bit_index returns the index of the lowest set bit.
fn lowest_bit_index(mask int) int {
	mut i := 0
	mut m := mask
	for m & 1 == 0 {
		m >>= 1
		i++
	}
	return i
}

// blade_prod multiplies two basis blades given as bitmasks over orthonormal
// basis vectors with squared lengths `sig` (+1/-1, 0 for nilpotent).
// Returns the resulting mask and sign.
fn blade_prod(mask_a int, mask_b int, sig []f32) (int, f32) {
	mut ma := mask_a
	mut mb := mask_b
	mut sign := f32(1.0)
	for mb != 0 {
		low := mb & (-mb)
		mb &= mb - 1
		j := lowest_bit_index(low)
		// crossings: number of original A-vectors with index > j (the
		// incoming factor must pass exactly those when slotting into place)
		above := bit_count(mask_a) - bit_count(mask_a & ((1 << (j + 1)) - 1))
		if ma & low != 0 {
			if above % 2 == 1 {
				sign = -sign * sig[j]
			} else {
				sign *= sig[j]
			}
			ma ^= low // remove the matched vector
		} else {
			if above % 2 == 1 {
				sign = -sign
			}
			ma |= low
		}
	}
	return ma, sign
}

// gen_clifford_table computes the structure-constant tensor T[l][k][j]
// (flat [d,d,d]) for Cl(p,q): e_k · e_j = sum_l T[l][k][j] e_l, where the
// basis index l is also the bitmask of the blade.
fn gen_clifford_table(p int, q int) []f32 {
	d := 1 << (p + q) // one entry per blade (bitmask), not per basis vector
	mut sig := []f32{len: d}
	for i in 0 .. d {
		if i < p {
			sig[i] = 1.0
		} else {
			sig[i] = -1.0
		}
	}
	mut t := []f32{len: d * d * d}
	for k in 0 .. d {
		for j in 0 .. d {
			mask, sign := blade_prod(k, j, sig)
			t[mask * d * d + k * d + j] += sign
		}
	}
	return t
}

// cga_points: embedding 3-d points into Cl(4,1).  Basis components use the
// 5 vectors e1..e5 (metrics + + + + -) with the standard null basis
// e∞ = e5 + e4, e0 = (e5 - e4) / 2.  The conformal point is
// p̂ = p + ½p² e∞ + e0, whose blade index 0 is the scalar.

// conformal_point embeds [n, 3] euclidean points into [n, 32] conformal
// points.  Blade layout uses bitmask indexing: component k is the blade with
// mask k (e1=1, e2=2, e3=4, e4=8, e5=16).  p̂ = p + ½p² e∞ + e0 with
// e∞ = e5+e4 (indices 8 and 16) and e0 = (e5-e4)/2 (index 8: -1/2,
// index 16: +1/2).
fn conformal_point(p mlx.Array) mlx.Array {
	n := p.dim(0)
	p2 := mlx.s_mul(p.square().sum_axis(1, false), 0.5) // [n]
	mut cols := []mlx.Array{cap: 32}
	cols << mlx.zeros([n, 1], .float32) // 0 scalar
	cols << p.take_axis(mlx.array_i32([i32(0)], [1]), 1) // 1 (p1)
	cols << p.take_axis(mlx.array_i32([i32(1)], [1]), 1) // 2 (p2)
	cols << mlx.zeros([n, 1], .float32) // 3
	cols << p.take_axis(mlx.array_i32([i32(2)], [1]), 1) // 4 (p3)
	cols << mlx.zeros([n, 1], .float32) // 5
	cols << mlx.zeros([n, 1], .float32) // 6
	cols << mlx.zeros([n, 1], .float32) // 7
	cols << p2.reshape([n, 1]).subtract(mlx.f32_scalar(0.5)) // 8 (e4: ½p²-½)
	for i in 0 .. 7 {
		cols << mlx.zeros([n, 1], .float32) // 9..15
	}
	cols << p2.reshape([n, 1]).add(mlx.f32_scalar(0.5)) // 16 (e5: ½p²+½)
	for i in 0 .. 15 {
		cols << mlx.zeros([n, 1], .float32) // 17..31
	}
	return mlx.concatenate(cols, 1)
}

// extract_point normalises a scaled conformal point back to euclidean:
// given x = λ·p̂, the e0 part shows at mask 8 (λ = -2·x[8]) and the three
// euclidean components are at masks 1, 2, 4.
fn extract_point(x mlx.Array) (mlx.Array, mlx.Array) {
	n := x.dim(0)
	// λ = x[16] - x[8] (e∞ contributes the same p² term to both masks while
	// e0 flips its sign between them)
	lam := x.take_axis(mlx.array_i32([i32(16)], [1]), 1).subtract(x.take_axis(mlx.array_i32([
		i32(8),
	], [1]), 1)) // [n,1]
	eu := x.take_axis(mlx.array_i32([i32(1), 2, 4], [3]), 1).divide(lam)
	return eu, lam
}

// bit_count_pub exposes the popcount for host testing helpers.
pub fn bit_count_pub(mask int) int {
	return bit_count(mask)
}
