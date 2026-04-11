#pragma once
#include "common.h"
#include "utils.h"

struct IValueFlat;
struct IValueNode;

/// Load a shared library (.so) that registers PyTorch ops via TORCH_LIBRARY.
/// The library is loaded with RTLD_GLOBAL so registered ops are visible to
/// the c10::Dispatcher. Returns true on success.
bool load_torch_library(rust::String path);

/// Call a PyTorch op by name through the c10::Dispatcher.
///
/// `op_name` is the operator name, e.g. "torchvision::roi_align"
/// `overload` is the overload name, e.g. "default" or ""
/// `args` is a flattened IValue tree encoding the positional arguments
///   in schema order. Each top-level node becomes one IValue on the stack.
///
/// Returns the result as a flattened IValue tree.
IValueFlat dispatch_op(
    rust::String op_name,
    rust::String overload,
    IValueFlat args);

/// List all ops currently registered with the dispatcher that match a
/// namespace prefix (e.g. "torchvision"). Returns fully-qualified names.
rust::Vec<rust::String> list_registered_ops(rust::String ns_prefix);

/// Execute an entire computation graph in a single C++ call.
///
/// The graph is encoded as a flat instruction stream using IValueNode with
/// extended tags:
///   tag=20: BEGIN_OP   — string_val=target, child_count=num_args
///   tag=21: OUTPUT_NAME — string_val=output name for this op result
///   tag=22: OVERLOAD    — string_val=overload name (follows BEGIN_OP)
///   tag=10: TENSOR_REF  — string_val=name to look up in values map
///   tag=0-9: regular IValue types (inline literals)
///   tag=7:  LIST with child_count=N (next N items are list elements)
///
/// `graph` is the instruction stream.
/// `initial_names` / `initial_tensors` are the pre-populated values map
///   (weights + user inputs).
/// `output_names` specifies which values to return.
///
/// Returns the selected outputs as a flattened IValue tree.
IValueFlat execute_graph(
    rust::Vec<IValueNode> graph,
    rust::Vec<rust::String> initial_names,
    TensorList initial_tensors,
    rust::Vec<rust::String> output_names);
