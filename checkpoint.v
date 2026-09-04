module nn

import os
import x.json2
import mlx

// checkpoint.v — loading pretrained checkpoints (safetensors) whose tensor
// names and layouts follow other frameworks' conventions.
//
// A Checkpoint wraps the tensor map plus the safetensors JSON header (names,
// shapes, dtypes) so missing keys and shape mismatches can be reported
// before anything is assigned.  LoadRule maps one checkpoint tensor onto one
// internal parameter key, with an optional axis permutation for layout
// conversion (PyTorch conv weights are [out, in, kH, kW] NCHW-style while
// MLX conv2d wants [out, kH, kW, in]; PyTorch linear weights are [out, in]
// while ours are [in, out]).

// LoadRule maps checkpoint tensor `from` onto internal parameter key `to`,
// applying `perm` (axis permutation) when non-empty.
pub struct LoadRule {
pub:
	from string
	to   string
	perm []int
}

// torch_conv_rule maps a PyTorch conv weight [out, in, kH, kW] onto our
// conv2d weight [out, kH, kW, in].
pub fn torch_conv_rule(from string, to string) LoadRule {
	return LoadRule{
		from: from
		to: to
		perm: [0, 2, 3, 1]
	}
}

// torch_linear_rule maps a PyTorch linear weight [out, in] onto our linear
// weight [in, out].
pub fn torch_linear_rule(from string, to string) LoadRule {
	return LoadRule{
		from: from
		to: to
		perm: [1, 0]
	}
}

// plain_rule maps a tensor without layout conversion.
pub fn plain_rule(from string, to string) LoadRule {
	return LoadRule{
		from: from
		to: to
	}
}

// Checkpoint is an opened safetensors file: the tensor map plus the parsed
// header (name -> shape).
pub struct Checkpoint {
pub:
	path string
mut:
	tensors mlx.MapStringToArray
	shapes  map[string][]int
}

// open_checkpoint loads a safetensors file and parses its header.
pub fn open_checkpoint(path string) Checkpoint {
	tensors, _ := mlx.load_safetensors(path)
	return Checkpoint{
		path: path
		tensors: tensors
		shapes: parse_safetensors_header(path)
	}
}

// keys returns the tensor names recorded in the header (excluding metadata).
pub fn (c Checkpoint) keys() []string {
	mut out := []string{}
	for k, _ in c.shapes {
		out << k
	}
	return out
}

// shape_of returns the declared shape of tensor `name` ([]int if absent).
pub fn (c Checkpoint) shape_of(name string) []int {
	return c.shapes[name] or { []int{} }
}

// has reports whether tensor `name` exists in the checkpoint.
pub fn (c Checkpoint) has(name string) bool {
	return name in c.shapes
}

// tensor returns the named tensor, optionally permuted, materialised on the
// CPU stream (safetensors loads are lazy).
pub fn (c Checkpoint) tensor(name string, perm []int) mlx.Array {
	if !c.has(name) {
		panic('nn: checkpoint ${c.path} has no tensor `${name}`; available: ${c.keys()}')
	}
	mut t := c.tensors.get(name)
	if perm.len > 0 {
		t = t.transpose_axes(perm)
	}
	t.eval()
	return t
}

// close releases the tensor map.
pub fn (mut c Checkpoint) close() {
	c.tensors.free()
}

// parse_safetensors_header reads the JSON header of a safetensors file and
// returns name -> shape.  The header layout is: 8-byte little-endian length,
// then a JSON object {name: {dtype, shape, data_offsets}}.
fn parse_safetensors_header(path string) map[string][]int {
	mut f := os.open(path) or { panic('nn: cannot open ${path}: ${err}') }
	defer {
		f.close()
	}
	mut len_buf := []u8{len: 8}
	f.read_bytes_into(0, mut len_buf) or { panic('nn: cannot read ${path}: ${err}') }
	mut header_len := u64(0)
	for i in 0 .. 8 {
		header_len |= u64(len_buf[i]) << (8 * i)
	}
	header_bytes := f.read_bytes_at(int(header_len), 8)
	header := json2.decode[map[string]json2.Any](header_bytes.bytestr()) or {
		panic('nn: bad safetensors header in ${path}: ${err}')
	}
	mut shapes := map[string][]int{}
	for name, entry in header {
		if name == '__metadata__' {
			continue
		}
		fields := entry.as_map()
		shape_any := fields['shape'] or { continue }
		mut dims := []int{}
		for d in shape_any.as_array() {
			dims << d.int()
		}
		shapes[name] = dims
	}
	return shapes
}
