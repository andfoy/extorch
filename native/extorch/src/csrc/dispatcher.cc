#include "extorch/src/native.rs.h"
#include "extorch/include/dispatcher.h"
#include "extorch/include/ivalue_utils.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <dlfcn.h>

// ============================================================================
// Library loading
// ============================================================================

bool load_torch_library(rust::String path) {
    std::string path_str(path);
    void *handle = dlopen(path_str.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        throw std::runtime_error(
            std::string("Failed to load library: ") + dlerror());
    }
    return true;
}

// ============================================================================
// IValue reconstruction from flat representation
// ============================================================================

// Tags mirror jit.cc flatten_ivalue_recursive:
//   0=tensor, 1=int, 2=float, 3=bool, 4=string, 5=none,
//   6=tuple, 7=list, 8=dict

static c10::IValue unflatten_node(
    const rust::Vec<IValueNode> &nodes,
    size_t &idx)
{
    if (idx >= nodes.size()) {
        throw std::runtime_error("IValueFlat: unexpected end of nodes");
    }
    const auto &node = nodes[idx];
    idx++;

    switch (node.tag) {
    case 0: // tensor
        return c10::IValue(*node.tensor);
    case 1: // int
        return c10::IValue(node.int_val);
    case 2: // float
        return c10::IValue(node.float_val);
    case 3: // bool
        return c10::IValue(node.bool_val);
    case 4: // string
        return c10::IValue(std::string(node.string_val));
    case 5: // none
        return c10::IValue();
    case 6: { // tuple
        std::vector<c10::IValue> elems;
        elems.reserve(static_cast<size_t>(node.child_count));
        for (int64_t i = 0; i < node.child_count; i++) {
            elems.push_back(unflatten_node(nodes, idx));
        }
        return c10::IValue(c10::ivalue::Tuple::create(std::move(elems)));
    }
    case 7: { // list
        // Peek ahead to determine the element type. If all children are
        // tensors, build a c10::List<at::Tensor> (TensorList) so the
        // dispatcher's boxed unboxing doesn't reject it as GenericList.
        bool all_tensors = true;
        {
            size_t peek = idx;
            for (int64_t i = 0; i < node.child_count && all_tensors; i++) {
                if (peek < nodes.size() && nodes[peek].tag != 0) {
                    all_tensors = false;
                }
                // Skip this node (only works for leaf nodes; containers
                // would need recursive skip, but tensor lists are always flat)
                peek++;
            }
        }

        if (all_tensors && node.child_count > 0) {
            c10::List<at::Tensor> tlist;
            tlist.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                tlist.push_back(unflatten_node(nodes, idx).toTensor());
            }
            return c10::IValue(std::move(tlist));
        } else {
            c10::impl::GenericList list(c10::AnyType::get());
            for (int64_t i = 0; i < node.child_count; i++) {
                list.push_back(unflatten_node(nodes, idx));
            }
            return c10::IValue(std::move(list));
        }
    }
    case 8: { // dict
        c10::impl::GenericDict dict(
            c10::AnyType::get(), c10::AnyType::get());
        int64_t n_pairs = node.child_count / 2;
        for (int64_t i = 0; i < n_pairs; i++) {
            auto key = unflatten_node(nodes, idx);
            auto val = unflatten_node(nodes, idx);
            dict.insert(std::move(key), std::move(val));
        }
        return c10::IValue(std::move(dict));
    }
    case 9: { // device — string_val holds type ("cuda"/"cpu"), int_val holds index
        std::string dev_str(node.string_val);
        if (node.int_val >= 0) {
            return c10::IValue(c10::Device(dev_str + ":" + std::to_string(node.int_val)));
        }
        return c10::IValue(c10::Device(dev_str));
    }
    default:
        throw std::runtime_error(
            "IValueFlat: unknown tag " + std::to_string(node.tag));
    }
}

// Unflatten a top-level IValueFlat into a vector of IValues.
// Each root-level node (parent_idx == -1) becomes one entry.
static std::vector<c10::IValue> unflatten_args(IValueFlat flat) {
    std::vector<c10::IValue> args;
    size_t idx = 0;
    while (idx < flat.nodes.size()) {
        args.push_back(unflatten_node(flat.nodes, idx));
    }
    return args;
}

// ============================================================================
// Generic op dispatch
// ============================================================================

IValueFlat dispatch_op(
    rust::String op_name,
    rust::String overload,
    IValueFlat args)
{
    std::string name_str(op_name);
    std::string overload_str(overload);

    // Look up the op schema in the dispatcher.
    auto &dispatcher = c10::Dispatcher::singleton();
    auto op_handle = dispatcher.findSchemaOrThrow(
        name_str.c_str(), overload_str.c_str());

    // Build the IValue stack from the flattened args.
    auto stack = unflatten_args(std::move(args));

    // Call the op through the boxed dispatch path.
    op_handle.callBoxed(&stack);

    // Pack the results. If there's a single return value, flatten it
    // directly. If multiple, wrap in a tuple first.
    if (stack.size() == 1) {
        return flatten_ivalue(stack[0]);
    } else {
        // Multiple returns → wrap as tuple
        return flatten_ivalue(c10::IValue(
            c10::ivalue::Tuple::create(std::move(stack))));
    }
}

// ============================================================================
// Op listing
// ============================================================================

rust::Vec<rust::String> list_registered_ops(rust::String ns_prefix) {
    std::string prefix(ns_prefix);
    rust::Vec<rust::String> result;

    auto &dispatcher = c10::Dispatcher::singleton();
    // Iterate all registered ops and filter by namespace prefix.
    for (const auto &op : dispatcher.getAllOpNames()) {
        auto full_name = op.name;
        if (prefix.empty() || full_name.substr(0, prefix.size()) == prefix) {
            std::string display = full_name;
            if (!op.overload_name.empty()) {
                display += "." + op.overload_name;
            }
            result.push_back(rust::String(display));
        }
    }
    return result;
}
