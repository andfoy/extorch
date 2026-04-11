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

// ============================================================================
// Compiled Graph — pre-resolved ops + integer-indexed values
// ============================================================================

/// Compile a graph instruction stream into an optimized C++ representation.
///
/// Pre-resolves all operator schemas, converts string refs to integer
/// indices, and pre-builds argument templates. The returned object
/// holds everything needed for forward passes with zero per-op overhead.
///
/// `graph` is the same instruction stream as execute_graph.
/// `value_names` are all tensor names that will be in the values map
///   (parameter names + user input names), in the order they will be
///   passed to run_compiled_graph.
/// `output_names` specifies which values to return after execution.
std::shared_ptr<CrossCompiledGraph> compile_graph(
    rust::Vec<IValueNode> graph,
    rust::Vec<rust::String> value_names,
    rust::Vec<rust::String> output_names);

/// Run a pre-compiled graph. Only passes tensors — all op resolution,
/// arg templates, and index mapping were done at compile time.
///
/// `tensors` must be in the same order as `value_names` passed to
/// compile_graph.
TensorList run_compiled_graph(
    const std::shared_ptr<CrossCompiledGraph> &compiled,
    TensorList tensors);

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
