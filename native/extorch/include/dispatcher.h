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
