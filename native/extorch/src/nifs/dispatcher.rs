use crate::encoding::jit::ivalue_flat_to_term;
use crate::native::torch;
use crate::shared_types::{TensorStruct, CompiledGraphStruct, Reference};

use cxx::SharedPtr;
use rustler::{Atom, Env, Error, NifResult, Term};

mod atoms {
    rustler::atoms! {
        tensor,
        int,
        float,
        bool_tag = "bool",
        list,
        none,
        nil,
        device,
        string,
        begin_op,
        overload,
        output,
        ref_tag = "ref",
        arg_name,
    }
}

/// Helper to convert a cxx error into a NifResult error.
fn cxx_err_to_nif(err: cxx::Exception) -> Error {
    let err_msg = err.what().to_owned();
    let err_parts: Vec<&str> = err_msg.split('\n').collect();
    Error::RaiseTerm(Box::new(err_parts[0].to_owned()))
}

/// Build an IValueFlat node for a tensor.
fn tensor_node(tensor: &SharedPtr<torch::CrossTensor>) -> torch::IValueNode {
    torch::IValueNode {
        tag: 0,
        tensor: tensor.clone(),
        int_val: 0,
        float_val: 0.0,
        bool_val: false,
        string_val: String::new(),
        parent_idx: -1,
        child_count: 0,
    }
}

/// Build an IValueFlat node for an integer.
fn int_node(val: i64) -> torch::IValueNode {
    torch::IValueNode {
        tag: 1,
        tensor: SharedPtr::null(),
        int_val: val,
        float_val: 0.0,
        bool_val: false,
        string_val: String::new(),
        parent_idx: -1,
        child_count: 0,
    }
}

/// Build an IValueFlat node for a float.
fn float_node(val: f64) -> torch::IValueNode {
    torch::IValueNode {
        tag: 2,
        tensor: SharedPtr::null(),
        int_val: 0,
        float_val: val,
        bool_val: false,
        string_val: String::new(),
        parent_idx: -1,
        child_count: 0,
    }
}

/// Build an IValueFlat node for a bool.
fn bool_node(val: bool) -> torch::IValueNode {
    torch::IValueNode {
        tag: 3,
        tensor: SharedPtr::null(),
        int_val: 0,
        float_val: 0.0,
        bool_val: val,
        string_val: String::new(),
        parent_idx: -1,
        child_count: 0,
    }
}

/// Build an IValueFlat node for None.
fn none_node() -> torch::IValueNode {
    torch::IValueNode {
        tag: 5,
        tensor: SharedPtr::null(),
        int_val: 0,
        float_val: 0.0,
        bool_val: false,
        string_val: String::new(),
        parent_idx: -1,
        child_count: 0,
    }
}

/// Build an IValueFlat node for a device.
fn device_node(device_type: &str, index: i64) -> torch::IValueNode {
    torch::IValueNode {
        tag: 9,
        tensor: SharedPtr::null(),
        int_val: index,
        float_val: 0.0,
        bool_val: false,
        string_val: device_type.to_string(),
        parent_idx: -1,
        child_count: 0,
    }
}

/// Build an IValueFlat node for a list header.
fn list_node(child_count: i64) -> torch::IValueNode {
    torch::IValueNode {
        tag: 7,
        tensor: SharedPtr::null(),
        int_val: 0,
        float_val: 0.0,
        bool_val: false,
        string_val: String::new(),
        parent_idx: -1,
        child_count,
    }
}

/// Load a shared library that registers PyTorch/torchvision ops.
#[rustler::nif]
pub fn load_torch_library(path: String) -> NifResult<bool> {
    torch::load_torch_library(path).map_err(cxx_err_to_nif)
}

/// Call an op by name through the c10::Dispatcher.
///
/// `args` is a list of tagged argument values:
///   {:tensor, tensor_struct} | {:int, integer} | {:float, float} |
///   {:bool, boolean} | :none | {:list, [tagged_args]}
///
/// Returns the result as an Elixir term (tensor, tuple, list, scalar, etc.)
#[rustler::nif]
pub fn dispatch_op<'a>(
    env: Env<'a>,
    op_name: String,
    overload: String,
    args: Vec<Term<'a>>,
) -> NifResult<Term<'a>> {
    let mut nodes: Vec<torch::IValueNode> = Vec::new();
    build_ivalue_nodes(env, &args, &mut nodes)?;

    let flat = torch::IValueFlat { nodes };
    let result = torch::dispatch_op(op_name, overload, flat).map_err(cxx_err_to_nif)?;
    Ok(ivalue_flat_to_term(env, &result))
}

/// List all registered ops matching a namespace prefix.
#[rustler::nif]
pub fn list_registered_ops(ns_prefix: String) -> NifResult<Vec<String>> {
    let ops = torch::list_registered_ops(ns_prefix).map_err(cxx_err_to_nif)?;
    Ok(ops.into_iter().map(|s| s.to_string()).collect())
}

/// Compile a graph into a pre-optimized C++ representation.
/// All op schemas are resolved, string refs converted to slot indices.
#[rustler::nif]
pub fn compile_graph<'a>(
    env: Env<'a>,
    graph: Vec<Term<'a>>,
    value_names: Vec<String>,
    output_names: Vec<String>,
) -> NifResult<CompiledGraphStruct<'a>> {
    let mut instructions: Vec<torch::IValueNode> = Vec::new();
    for term in &graph {
        decode_graph_instruction(env, *term, &mut instructions)?;
    }

    let compiled = torch::compile_graph(instructions, value_names, output_names)
        .map_err(cxx_err_to_nif)?;

    let wrapped = torch::CrossCompiledGraphRef { graph: compiled };
    let resource = rustler::ResourceArc::new(wrapped);
    Ok(CompiledGraphStruct {
        resource,
        reference: Reference::new(),
    })
}

/// Run a pre-compiled graph. Tensors in, tensors out — no encoding overhead.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn run_compiled_graph<'a>(
    compiled: CompiledGraphStruct<'a>,
    tensors: Vec<TensorStruct<'a>>,
) -> NifResult<Vec<TensorStruct<'a>>> {
    let tensor_list = make_tensor_list(&tensors);
    let result = torch::run_compiled_graph(&compiled.resource.graph, tensor_list)
        .map_err(cxx_err_to_nif)?;

    Ok(result.values.into_iter()
        .filter(|t| t.used)
        .map(|t| t.tensor.into())
        .collect())
}

/// Execute an entire computation graph in a single NIF call.
///
/// Eliminates per-node NIF boundary crossings by running the full graph
/// loop in C++. The graph is encoded as a flat instruction stream.
///
/// Args:
///   graph: list of tagged instructions (see build_graph_instructions)
///   initial_names: list of value map keys
///   initial_tensors: list of tensors for those keys
///   output_names: which values to return
#[rustler::nif(schedule = "DirtyCpu")]
pub fn execute_graph<'a>(
    env: Env<'a>,
    graph: Vec<Term<'a>>,
    initial_names: Vec<String>,
    initial_tensors: Vec<TensorStruct<'a>>,
    output_names: Vec<String>,
) -> NifResult<Term<'a>> {
    // Build the instruction stream
    let mut instructions: Vec<torch::IValueNode> = Vec::new();
    for term in &graph {
        decode_graph_instruction(env, *term, &mut instructions)?;
    }

    // Build tensor list for initial values
    let tensor_list = make_tensor_list(&initial_tensors);

    let result = torch::execute_graph(
        instructions,
        initial_names,
        tensor_list,
        output_names,
    ).map_err(cxx_err_to_nif)?;

    Ok(ivalue_flat_to_term(env, &result))
}

/// Build a TensorList from a slice of TensorStructs.
fn make_tensor_list(inputs: &[TensorStruct]) -> torch::TensorList {
    let values: Vec<torch::TensorOut> = inputs
        .iter()
        .map(|t| torch::TensorOut {
            tensor: t.resource.tensor.clone(),
            used: true,
        })
        .collect();
    torch::TensorList { values, used: true }
}

/// Decode a graph instruction term into IValueNode entries.
///
/// Instruction formats:
///   {:begin_op, target, num_args}
///   {:overload, name}
///   {:output, name}
///   {:ref, name}            — tensor reference (look up in values)
///   {:tensor, tensor_struct} — inline tensor
///   {:int, value}
///   {:float, value}
///   {:bool, value}
///   :none
///   {:list, [items]}
///   {:device, dev}
fn decode_graph_instruction<'a>(
    env: Env<'a>,
    term: Term<'a>,
    nodes: &mut Vec<torch::IValueNode>,
) -> NifResult<()> {
    // Try 3-tuple first: {:begin_op, target, num_args}
    if let Ok((tag, target, num_args)) = term.decode::<(Atom, String, i64)>() {
        if tag == atoms::begin_op() {
            nodes.push(torch::IValueNode {
                tag: 20,
                tensor: SharedPtr::null(),
                int_val: 0,
                float_val: 0.0,
                bool_val: false,
                string_val: target,
                parent_idx: -1,
                child_count: num_args,
            });
            return Ok(());
        }
    }

    // Try 2-tuple tagged values
    if let Ok((tag, value)) = term.decode::<(Atom, Term<'a>)>() {
        if tag == atoms::overload() {
            let name: String = value.decode()?;
            nodes.push(torch::IValueNode {
                tag: 22,
                tensor: SharedPtr::null(),
                int_val: 0,
                float_val: 0.0,
                bool_val: false,
                string_val: name,
                parent_idx: -1,
                child_count: 0,
            });
        } else if tag == atoms::output() {
            let name: String = value.decode()?;
            nodes.push(torch::IValueNode {
                tag: 21,
                tensor: SharedPtr::null(),
                int_val: 0,
                float_val: 0.0,
                bool_val: false,
                string_val: name,
                parent_idx: -1,
                child_count: 0,
            });
        } else if tag == atoms::ref_tag() {
            let name: String = value.decode()?;
            nodes.push(torch::IValueNode {
                tag: 10,
                tensor: SharedPtr::null(),
                int_val: 0,
                float_val: 0.0,
                bool_val: false,
                string_val: name,
                parent_idx: -1,
                child_count: 0,
            });
        } else if tag == atoms::arg_name() {
            // Tag 23: argument name for schema-aware reordering
            let name: String = value.decode()?;
            nodes.push(torch::IValueNode {
                tag: 23,
                tensor: SharedPtr::null(),
                int_val: 0,
                float_val: 0.0,
                bool_val: false,
                string_val: name,
                parent_idx: -1,
                child_count: 0,
            });
        } else {
            // Delegate to the regular arg decoder for tensor/int/float/bool/list/device
            decode_arg_to_nodes(env, term, nodes)?;
        }
        return Ok(());
    }

    // Bare atom (:none)
    decode_arg_to_nodes(env, term, nodes)?;
    Ok(())
}

/// Recursively convert Elixir tagged arg terms into IValueNode entries.
fn build_ivalue_nodes<'a>(
    env: Env<'a>,
    args: &[Term<'a>],
    nodes: &mut Vec<torch::IValueNode>,
) -> NifResult<()> {
    for arg in args {
        decode_arg_to_nodes(env, *arg, nodes)?;
    }
    Ok(())
}

fn decode_arg_to_nodes<'a>(
    env: Env<'a>,
    term: Term<'a>,
    nodes: &mut Vec<torch::IValueNode>,
) -> NifResult<()> {
    // Try to decode as a tagged tuple: {:tensor, t}, {:int, n}, etc.
    if let Ok((tag, value)) = term.decode::<(Atom, Term<'a>)>() {
        if tag == atoms::tensor() {
            let t: TensorStruct<'a> = value.decode()?;
            nodes.push(tensor_node(&t.resource.tensor));
        } else if tag == atoms::int() {
            let v: i64 = value.decode()?;
            nodes.push(int_node(v));
        } else if tag == atoms::float() {
            let v: f64 = value.decode()?;
            nodes.push(float_node(v));
        } else if tag == atoms::bool_tag() {
            let v: bool = value.decode()?;
            nodes.push(bool_node(v));
        } else if tag == atoms::list() {
            let items: Vec<Term<'a>> = value.decode()?;
            nodes.push(list_node(items.len() as i64));
            build_ivalue_nodes(env, &items, nodes)?;
        } else if tag == atoms::device() {
            // {:device, "cuda"} or {:device, {"cuda", 0}}
            if let Ok((dev_type, dev_idx)) = value.decode::<(String, i64)>() {
                nodes.push(device_node(&dev_type, dev_idx));
            } else if let Ok(dev_type) = value.decode::<String>() {
                nodes.push(device_node(&dev_type, -1));
            } else {
                return Err(Error::RaiseTerm(Box::new(
                    "device arg must be a string or {string, index} tuple".to_owned(),
                )));
            }
        } else if tag == atoms::string() {
            let v: String = value.decode()?;
            nodes.push(torch::IValueNode {
                tag: 4,
                tensor: SharedPtr::null(),
                int_val: 0,
                float_val: 0.0,
                bool_val: false,
                string_val: v,
                parent_idx: -1,
                child_count: 0,
            });
        } else if tag == atoms::ref_tag() {
            // Tensor reference — used inside list args in execute_graph
            let name: String = value.decode()?;
            nodes.push(torch::IValueNode {
                tag: 10,
                tensor: SharedPtr::null(),
                int_val: 0,
                float_val: 0.0,
                bool_val: false,
                string_val: name,
                parent_idx: -1,
                child_count: 0,
            });
        } else {
            return Err(Error::RaiseTerm(Box::new(
                "Unknown dispatch arg tag".to_owned(),
            )));
        }
    } else if let Ok(atom) = term.decode::<Atom>() {
        if atom == atoms::none() || atom == atoms::nil() {
            nodes.push(none_node());
        } else {
            return Err(Error::RaiseTerm(Box::new(
                "Unknown dispatch arg atom".to_owned(),
            )));
        }
    } else {
        return Err(Error::RaiseTerm(Box::new(
            "dispatch_op args must be tagged tuples like {:tensor, t} or {:int, 42}".to_owned(),
        )));
    }
    Ok(())
}
