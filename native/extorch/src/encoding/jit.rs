use crate::native::torch;
use crate::shared_types::{JitModuleStruct, Reference, TensorStruct};

use cxx::SharedPtr;
use rustler::{Encoder, Env, Term, ResourceArc};
use rustler::types::tuple::make_tuple;

impl<'a> From<SharedPtr<torch::CrossModule>> for JitModuleStruct<'a> {
    fn from(value: SharedPtr<torch::CrossModule>) -> Self {
        let wrapped_module = torch::CrossModuleRef { module: value };
        let resource = ResourceArc::new(wrapped_module);
        let reference = Reference::new();
        // Default device - will be set by the calling NIF
        let device = torch::Device {
            device: "cpu".to_string(),
            index: -1,
        };
        JitModuleStruct {
            resource,
            reference,
            device,
        }
    }
}

/// Reconstruct a flat IValue tree into nested Elixir terms.
///
/// The IValueFlat contains nodes in pre-order traversal. Each node has:
/// - tag: type discriminator
/// - parent_idx: index of parent (-1 for root)
/// - child_count: number of children (for containers)
///
/// We reconstruct by walking through the nodes and building Elixir terms.
pub fn ivalue_flat_to_term<'a>(env: Env<'a>, flat: &torch::IValueFlat) -> Term<'a> {
    if flat.nodes.is_empty() {
        rustler::types::atom::nil().encode(env)
    } else {
        ivalue_node_to_term(env, &flat.nodes, 0).0
    }
}

/// Recursively convert a node and its children to an Elixir term.
/// Returns (term, next_index) where next_index is the index after this subtree.
fn ivalue_node_to_term<'a>(
    env: Env<'a>,
    nodes: &[torch::IValueNode],
    idx: usize,
) -> (Term<'a>, usize) {
    let node = &nodes[idx];
    match node.tag {
        // Tensor
        0 => {
            let tensor_struct: TensorStruct<'a> = node.tensor.clone().into();
            (tensor_struct.encode(env), idx + 1)
        }
        // Int
        1 => (node.int_val.encode(env), idx + 1),
        // Float
        2 => (node.float_val.encode(env), idx + 1),
        // Bool
        3 => (node.bool_val.encode(env), idx + 1),
        // String
        4 => {
            let s: &str = node.string_val.as_str();
            (s.encode(env), idx + 1)
        }
        // None
        5 => (rustler::types::atom::nil().encode(env), idx + 1),
        // Tuple
        6 => {
            let child_count = node.child_count as usize;
            let mut children: Vec<Term<'a>> = Vec::with_capacity(child_count);
            let mut next_idx = idx + 1;
            for _ in 0..child_count {
                let (child_term, new_idx) = ivalue_node_to_term(env, nodes, next_idx);
                children.push(child_term);
                next_idx = new_idx;
            }
            (make_tuple(env, &children), next_idx)
        }
        // List
        7 => {
            let child_count = node.child_count as usize;
            let mut children: Vec<Term<'a>> = Vec::with_capacity(child_count);
            let mut next_idx = idx + 1;
            for _ in 0..child_count {
                let (child_term, new_idx) = ivalue_node_to_term(env, nodes, next_idx);
                children.push(child_term);
                next_idx = new_idx;
            }
            (children.encode(env), next_idx)
        }
        // Dict (children are key-value pairs: key1, val1, key2, val2, ...)
        8 => {
            let child_count = node.child_count as usize; // This is num_entries * 2
            let num_entries = child_count / 2;
            let map = rustler::Term::map_new(env);
            let mut next_idx = idx + 1;
            let mut result_map = map;
            for _ in 0..num_entries {
                let (key_term, key_next) = ivalue_node_to_term(env, nodes, next_idx);
                let (val_term, val_next) = ivalue_node_to_term(env, nodes, key_next);
                result_map = result_map.map_put(key_term, val_term).unwrap_or(result_map);
                next_idx = val_next;
            }
            (result_map, next_idx)
        }
        // Unknown - encode as nil
        _ => (rustler::types::atom::nil().encode(env), idx + 1),
    }
}

/// Convert a Vec<NamedTensor> into an Elixir list of {name, tensor} tuples.
pub fn named_tensors_to_term<'a>(env: Env<'a>, named: &[torch::NamedTensor]) -> Term<'a> {
    let tuples: Vec<Term<'a>> = named
        .iter()
        .map(|nt| {
            let name: &str = nt.name.as_str();
            let tensor_struct: TensorStruct<'a> = nt.tensor.clone().into();
            make_tuple(env, &[name.encode(env), tensor_struct.encode(env)])
        })
        .collect();
    tuples.encode(env)
}
