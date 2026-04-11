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
    case 7: { // list — create typed lists for the dispatcher
        if (node.child_count == 0) {
            {
                std::vector<int64_t> empty;
                return c10::IValue(std::move(empty));
            }
        }

        // Peek at first child to determine type, assume homogeneous
        int64_t first_tag = (idx < nodes.size()) ? nodes[idx].tag : -1;
        bool homogeneous = true;
        {
            size_t peek = idx;
            for (int64_t i = 0; i < node.child_count && homogeneous; i++) {
                if (peek < nodes.size() && nodes[peek].tag != first_tag) {
                    homogeneous = false;
                }
                peek++;
            }
        }

        if (homogeneous && first_tag == 0) {
            c10::List<at::Tensor> tlist;
            tlist.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                tlist.push_back(unflatten_node(nodes, idx).toTensor());
            }
            return c10::IValue(std::move(tlist));
        } else if (homogeneous && first_tag == 1) {
            std::vector<int64_t> ivec;
            ivec.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                ivec.push_back(unflatten_node(nodes, idx).toInt());
            }
            return c10::IValue(std::move(ivec));
        } else if (homogeneous && first_tag == 2) {
            c10::List<double> flist;
            flist.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                flist.push_back(unflatten_node(nodes, idx).toDouble());
            }
            return c10::IValue(std::move(flist));
        } else if (homogeneous && first_tag == 3) {
            c10::List<bool> blist;
            blist.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                blist.push_back(unflatten_node(nodes, idx).toBool());
            }
            return c10::IValue(std::move(blist));
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

    // Look up the op schema in the dispatcher. Try the given overload
    // first; fall back to empty overload if "default" is not found
    // (torch.export uses ".default" but many ops register without one).
    auto &dispatcher = c10::Dispatcher::singleton();
    auto schema = dispatcher.findSchema({name_str.c_str(), overload_str.c_str()});
    if (!schema.has_value() && overload_str == "default") {
        schema = dispatcher.findSchema({name_str.c_str(), ""});
    }
    if (!schema.has_value()) {
        throw std::runtime_error(
            "dispatch_op: no schema for " + name_str + "." + overload_str);
    }

    // Build the IValue stack from the flattened args.
    auto stack = unflatten_args(std::move(args));

    // Call the op through the boxed dispatch path.
    schema->callBoxed(&stack);

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

// ============================================================================
// Graph executor — run an entire computation graph in a single C++ call
// ============================================================================

// Extended tags for the graph instruction stream:
constexpr int64_t TAG_TENSOR_REF = 10;   // Look up tensor by name in values map
constexpr int64_t TAG_BEGIN_OP   = 20;   // Start of an op: string_val=target, child_count=num_args
constexpr int64_t TAG_OUTPUT     = 21;   // Output name: string_val=name
constexpr int64_t TAG_OVERLOAD   = 22;   // Overload name: string_val=overload

// Read one argument from the instruction stream.
// Regular IValue tags (0-9) are handled as literals.
// TAG_TENSOR_REF (10) looks up the name in the values map.
// TAG 7 (list) reads child_count sub-arguments.
static c10::IValue read_arg(
    const rust::Vec<IValueNode> &graph,
    size_t &pc,
    const std::unordered_map<std::string, c10::IValue> &values)
{
    const auto &node = graph[pc];
    pc++;

    switch (node.tag) {
    case 0: // inline tensor
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
    case 9: { // device
        std::string dev_str(node.string_val);
        if (node.int_val >= 0) {
            return c10::IValue(c10::Device(dev_str + ":" + std::to_string(node.int_val)));
        }
        return c10::IValue(c10::Device(dev_str));
    }
    case TAG_TENSOR_REF: { // look up tensor in values map
        std::string ref(node.string_val);
        auto it = values.find(ref);
        if (it == values.end()) {
            throw std::runtime_error("execute_graph: unknown ref '" + ref + "'");
        }
        return it->second;
    }
    case 7: { // list — create typed lists for the dispatcher
        if (node.child_count == 0) {
            // Empty list — the dispatcher typically expects int[] for
            // empty padding/stride args.
            {
                std::vector<int64_t> empty;
                return c10::IValue(std::move(empty));
            }
        }

        // Peek at child tags to determine element type
        int64_t first_tag = graph[pc].tag;
        bool homogeneous = true;
        for (int64_t i = 1; i < node.child_count && homogeneous; i++) {
            if (graph[pc + i].tag != first_tag) {
                homogeneous = false;
            }
        }

        if (homogeneous && (first_tag == 0 || first_tag == TAG_TENSOR_REF)) {
            c10::List<at::Tensor> tlist;
            tlist.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                tlist.push_back(read_arg(graph, pc, values).toTensor());
            }
            return c10::IValue(std::move(tlist));
        } else if (homogeneous && first_tag == 1) {
            // Read raw int values and construct IntList via explicit
            // c10::impl::GenericList with IntType element type.
            auto ilist = c10::impl::GenericList(c10::IntType::get());
            ilist.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                const auto &child = graph[pc];
                pc++;
                ilist.emplace_back(child.int_val);
            }
            return c10::IValue(std::move(ilist));
        } else if (homogeneous && first_tag == 2) {
            c10::List<double> flist;
            flist.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                flist.push_back(read_arg(graph, pc, values).toDouble());
            }
            return c10::IValue(std::move(flist));
        } else if (homogeneous && first_tag == 3) {
            c10::List<bool> blist;
            blist.reserve(static_cast<size_t>(node.child_count));
            for (int64_t i = 0; i < node.child_count; i++) {
                blist.push_back(read_arg(graph, pc, values).toBool());
            }
            return c10::IValue(std::move(blist));
        } else {
            c10::impl::GenericList list(c10::AnyType::get());
            for (int64_t i = 0; i < node.child_count; i++) {
                list.push_back(read_arg(graph, pc, values));
            }
            return c10::IValue(std::move(list));
        }
    }
    default:
        throw std::runtime_error(
            "execute_graph: unknown arg tag " + std::to_string(node.tag));
    }
}

IValueFlat execute_graph(
    rust::Vec<IValueNode> graph,
    rust::Vec<rust::String> initial_names,
    TensorList initial_tensors,
    rust::Vec<rust::String> output_names)
{
    // Build the initial values map from names + tensors
    auto tensors = unpack_tensor_list(std::move(initial_tensors));
    std::unordered_map<std::string, c10::IValue> values;
    values.reserve(initial_names.size());
    for (size_t i = 0; i < initial_names.size() && i < tensors.size(); i++) {
        values[std::string(initial_names[i])] = c10::IValue(tensors[i]);
    }

    auto &dispatcher = c10::Dispatcher::singleton();
    size_t pc = 0;
    size_t op_count = 0;
    const size_t max_ops = graph.size(); // safety bound

    while (pc < graph.size()) {
        if (++op_count > max_ops) {
            throw std::runtime_error(
                "execute_graph: exceeded max op count (" + std::to_string(max_ops) +
                "), likely infinite loop at pc=" + std::to_string(pc));
        }

        const auto &inst = graph[pc];

        if (inst.tag != TAG_BEGIN_OP) {
            throw std::runtime_error(
                "execute_graph: expected BEGIN_OP (tag=20) at pc=" + std::to_string(pc) +
                " but got tag=" + std::to_string(inst.tag));
        }

        std::string target(inst.string_val);
        int64_t num_args = inst.child_count;
        size_t pc_before = pc;
        pc++;

        // Read overload (must follow BEGIN_OP)
        std::string overload;
        if (pc < graph.size() && graph[pc].tag == TAG_OVERLOAD) {
            overload = std::string(graph[pc].string_val);
            pc++;
        }

        // Collect output names (follow overload)
        std::vector<std::string> out_names;
        while (pc < graph.size() && graph[pc].tag == TAG_OUTPUT) {
            out_names.push_back(std::string(graph[pc].string_val));
            pc++;
        }

        // Read arguments from instruction stream.
        // Args may be preceded by ARG_NAME (tag=23) for schema-aware
        // positional reordering.
        constexpr int64_t TAG_ARG_NAME = 23;
        std::vector<std::pair<std::string, c10::IValue>> named_args;
        named_args.reserve(static_cast<size_t>(num_args));
        for (int64_t i = 0; i < num_args; i++) {
            std::string arg_name;
            if (pc < graph.size() && graph[pc].tag == TAG_ARG_NAME) {
                arg_name = std::string(graph[pc].string_val);
                pc++;
            }
            named_args.push_back({arg_name, read_arg(graph, pc, values)});
        }

        // Resolve schema for arg reordering and dispatch.
        auto schema = dispatcher.findSchema({target.c_str(), overload.c_str()});
        if (!schema.has_value() && overload == "default") {
            schema = dispatcher.findSchema({target.c_str(), ""});
        }
        if (!schema.has_value()) {
            throw std::runtime_error(
                "execute_graph: no schema for " + target + "." + overload);
        }

        // Reorder args to match schema parameter positions.
        const auto &schema_params = schema->schema().arguments();
        std::vector<c10::IValue> args;
        args.reserve(schema_params.size());

        size_t named_idx = 0;
        for (size_t si = 0; si < schema_params.size(); si++) {
            const auto &param = schema_params[si];
            // Try to match by name
            bool found = false;
            if (named_idx < named_args.size()) {
                if (named_args[named_idx].first.empty() ||
                    named_args[named_idx].first == param.name()) {
                    args.push_back(std::move(named_args[named_idx].second));
                    named_idx++;
                    found = true;
                } else {
                    // Names don't match — look for this param in remaining named args
                    for (size_t ni = named_idx; ni < named_args.size(); ni++) {
                        if (named_args[ni].first == param.name()) {
                            args.push_back(std::move(named_args[ni].second));
                            // Shift remaining
                            named_args.erase(named_args.begin() + ni);
                            found = true;
                            break;
                        }
                    }
                }
            }
            if (!found) {
                // Use schema default
                if (param.default_value().has_value()) {
                    args.push_back(param.default_value().value());
                }
                // else: skip (will fail at dispatch if required)
            }
        }

        // Type coercion: adapt IValue types to match the schema.
        const auto &schema_args = schema->schema().arguments();

        for (size_t ai = 0; ai < args.size() && ai < schema_args.size(); ai++) {
            const auto &expected = schema_args[ai].type();
            if (expected->isSubtypeOf(*c10::TensorType::get())) {
                if (args[ai].isDouble()) {
                    args[ai] = c10::IValue(at::scalar_to_tensor(args[ai].toDouble()));
                } else if (args[ai].isInt()) {
                    args[ai] = c10::IValue(at::scalar_to_tensor(args[ai].toInt()));
                } else if (args[ai].isBool()) {
                    args[ai] = c10::IValue(at::scalar_to_tensor(args[ai].toBool()));
                }
            }
            if (expected->kind() == c10::TypeKind::OptionalType) {
                auto inner = expected->cast<c10::OptionalType>()->getElementType();
                if (inner->isSubtypeOf(*c10::DeviceObjType::get()) && args[ai].isDevice()) {
                    args[ai] = c10::IValue(args[ai].toDevice());
                }
            }
        }

        // Fill in missing arguments with schema defaults.
        while (args.size() < schema_args.size()) {
            size_t idx = args.size();
            const auto &arg = schema_args[idx];
            if (arg.default_value().has_value()) {
                args.push_back(arg.default_value().value());
            } else {
                // No default — this is a required arg that's missing
                throw std::runtime_error(
                    "execute_graph: missing required arg '" +
                    arg.name() + "' for " + target);
            }
        }

        schema->callBoxed(&args);

        // Store results in the values map
        if (args.size() == 1 && out_names.size() == 1) {
            values[out_names[0]] = std::move(args[0]);
        } else if (args.size() == 1 && args[0].isTuple()) {
            // Multi-output op returns a tuple
            auto tuple = args[0].toTuple();
            for (size_t i = 0; i < out_names.size() && i < tuple->elements().size(); i++) {
                values[out_names[i]] = tuple->elements()[i];
            }
        } else {
            // Multiple stack returns
            for (size_t i = 0; i < out_names.size() && i < args.size(); i++) {
                values[out_names[i]] = std::move(args[i]);
            }
        }
    }

    // Collect requested outputs
    IValueFlat result;
    if (output_names.size() == 1) {
        std::string name(output_names[0]);
        auto it = values.find(name);
        if (it != values.end()) {
            result = flatten_ivalue(it->second);
        }
    } else {
        // Multiple outputs → return as tuple
        std::vector<c10::IValue> outputs;
        for (const auto &name : output_names) {
            std::string n(name);
            auto it = values.find(n);
            if (it != values.end()) {
                outputs.push_back(it->second);
            }
        }
        result = flatten_ivalue(c10::IValue(
            c10::ivalue::Tuple::create(std::move(outputs))));
    }
    return result;
}

// ============================================================================
// Compiled Graph — pre-resolved ops with integer-indexed arguments
// ============================================================================

// An argument descriptor: either a slot index (tensor ref), a literal
// constant, or a typed list.
struct ArgDesc {
    enum Kind { SLOT, LITERAL, TENSOR_LIST_SLOTS, INT_LIST_LITERAL,
                FLOAT_LIST_LITERAL, BOOL_LIST_LITERAL };
    Kind kind;
    size_t slot;                        // for SLOT
    c10::IValue literal;                // for LITERAL
    std::vector<size_t> slot_list;      // for TENSOR_LIST_SLOTS
    std::vector<int64_t> int_list;      // for INT_LIST_LITERAL
    std::vector<double> float_list;     // for FLOAT_LIST_LITERAL
    std::vector<bool> bool_list;        // for BOOL_LIST_LITERAL

    ArgDesc() : kind(LITERAL), slot(0) {}
};

struct CompiledOp {
    c10::OperatorHandle handle;
    std::vector<ArgDesc> args;
    std::vector<size_t> output_slots;
    size_t num_schema_args;
};

struct CrossCompiledGraphImpl {
    std::vector<CompiledOp> ops;
    size_t num_slots;
    std::vector<size_t> output_slots;

    std::vector<CrossTensor> run(std::vector<CrossTensor> initial_tensors) const {
        std::vector<c10::IValue> values(num_slots);
        for (size_t i = 0; i < initial_tensors.size(); i++) {
            values[i] = c10::IValue(std::move(initial_tensors[i]));
        }

        for (const auto &op : ops) {
            std::vector<c10::IValue> args;
            args.reserve(op.args.size());
            for (const auto &desc : op.args) {
                switch (desc.kind) {
                case ArgDesc::SLOT:
                    args.push_back(values[desc.slot]);
                    break;
                case ArgDesc::LITERAL:
                    args.push_back(desc.literal);
                    break;
                case ArgDesc::TENSOR_LIST_SLOTS: {
                    c10::List<at::Tensor> tlist;
                    tlist.reserve(desc.slot_list.size());
                    for (auto s : desc.slot_list)
                        tlist.push_back(values[s].toTensor());
                    args.push_back(c10::IValue(std::move(tlist)));
                    break;
                }
                case ArgDesc::INT_LIST_LITERAL:
                    args.push_back(c10::IValue(desc.int_list));
                    break;
                case ArgDesc::FLOAT_LIST_LITERAL: {
                    c10::List<double> flist;
                    for (auto v : desc.float_list) flist.push_back(v);
                    args.push_back(c10::IValue(std::move(flist)));
                    break;
                }
                case ArgDesc::BOOL_LIST_LITERAL: {
                    c10::List<bool> blist;
                    for (auto v : desc.bool_list) blist.push_back(v);
                    args.push_back(c10::IValue(std::move(blist)));
                    break;
                }
                }
            }

            // Type coercion: adapt IValue types to match the schema.
            const auto &schema_args = op.handle.schema().arguments();
            for (size_t ai = 0; ai < args.size() && ai < schema_args.size(); ai++) {
                const auto &expected = schema_args[ai].type();
                // Scalar→Tensor for ops like aten::mul.Tensor
                if (expected->isSubtypeOf(*c10::TensorType::get())) {
                    if (args[ai].isDouble()) {
                        args[ai] = c10::IValue(at::scalar_to_tensor(args[ai].toDouble()));
                    } else if (args[ai].isInt()) {
                        args[ai] = c10::IValue(at::scalar_to_tensor(args[ai].toInt()));
                    } else if (args[ai].isBool()) {
                        args[ai] = c10::IValue(at::scalar_to_tensor(args[ai].toBool()));
                    }
                }
                // Wrap non-optional in Optional where schema expects it
                if (expected->kind() == c10::TypeKind::OptionalType) {
                    auto inner = expected->cast<c10::OptionalType>()->getElementType();
                    // Device → Optional<Device>
                    if (inner->isSubtypeOf(*c10::DeviceObjType::get()) && args[ai].isDevice()) {
                        args[ai] = c10::IValue(args[ai].toDevice());
                    }
                    // Int → Optional<ScalarType> (ScalarType is enum-backed by int)
                    if (inner->isSubtypeOf(*c10::IntType::get()) && args[ai].isInt()) {
                        // Already correct — int matches Optional<int>
                    }
                }
            }

            // Fill schema defaults
            while (args.size() < op.num_schema_args) {
                size_t idx = args.size();
                if (idx < schema_args.size() &&
                    schema_args[idx].default_value().has_value()) {
                    args.push_back(schema_args[idx].default_value().value());
                } else {
                    break;
                }
            }

            op.handle.callBoxed(&args);

            // Store results by slot index
            if (args.size() == 1 && op.output_slots.size() == 1) {
                values[op.output_slots[0]] = std::move(args[0]);
            } else if (args.size() == 1 && args[0].isTuple()) {
                auto tuple = args[0].toTuple();
                for (size_t i = 0; i < op.output_slots.size() &&
                     i < tuple->elements().size(); i++) {
                    values[op.output_slots[i]] = tuple->elements()[i];
                }
            } else {
                for (size_t i = 0; i < op.output_slots.size() &&
                     i < args.size(); i++) {
                    values[op.output_slots[i]] = std::move(args[i]);
                }
            }
        }

        std::vector<CrossTensor> result;
        result.reserve(output_slots.size());
        for (auto s : output_slots)
            result.push_back(values[s].toTensor());
        return result;
    }
};

static c10::OperatorHandle resolve_schema(
    c10::Dispatcher &dispatcher,
    const std::string &target,
    const std::string &overload)
{
    auto schema = dispatcher.findSchema({target.c_str(), overload.c_str()});
    if (!schema.has_value() && overload == "default") {
        schema = dispatcher.findSchema({target.c_str(), ""});
    }
    if (!schema.has_value()) {
        throw std::runtime_error("compile_graph: no schema for " + target + "." + overload);
    }
    return *schema;
}

std::shared_ptr<CrossCompiledGraph> compile_graph(
    rust::Vec<IValueNode> graph,
    rust::Vec<rust::String> value_names,
    rust::Vec<rust::String> output_names)
{
    auto compiled = std::make_shared<CrossCompiledGraphImpl>();

    std::unordered_map<std::string, size_t> name_to_slot;
    for (size_t i = 0; i < value_names.size(); i++) {
        name_to_slot[std::string(value_names[i])] = i;
    }
    size_t next_slot = value_names.size();

    auto &dispatcher = c10::Dispatcher::singleton();
    size_t pc = 0;

    while (pc < graph.size()) {
        if (graph[pc].tag != 20)
            throw std::runtime_error("compile_graph: expected BEGIN_OP at pc=" + std::to_string(pc));

        std::string target(graph[pc].string_val);
        int64_t num_args = graph[pc].child_count;
        pc++;

        std::string overload;
        if (pc < graph.size() && graph[pc].tag == 22) {
            overload = std::string(graph[pc].string_val);
            pc++;
        }

        std::vector<size_t> out_slots;
        while (pc < graph.size() && graph[pc].tag == 21) {
            std::string oname(graph[pc].string_val);
            size_t slot = next_slot++;
            name_to_slot[oname] = slot;
            out_slots.push_back(slot);
            pc++;
        }

        auto handle = resolve_schema(dispatcher, target, overload);

        // Read named args from instruction stream
        constexpr int64_t TAG_ARG_NAME_COMPILE = 23;
        std::vector<std::pair<std::string, ArgDesc>> named_descs;
        for (int64_t i = 0; i < num_args; i++) {
            std::string aname;
            if (pc < graph.size() && graph[pc].tag == TAG_ARG_NAME_COMPILE) {
                aname = std::string(graph[pc].string_val);
                pc++;
            }
            ArgDesc desc;
            const auto &node = graph[pc];

            switch (node.tag) {
            case 10: { // ref
                std::string ref(node.string_val);
                auto it = name_to_slot.find(ref);
                if (it == name_to_slot.end())
                    throw std::runtime_error("compile_graph: unknown ref '" + ref + "'");
                desc.kind = ArgDesc::SLOT;
                desc.slot = it->second;
                pc++;
                break;
            }
            case 0: desc.kind = ArgDesc::LITERAL; desc.literal = c10::IValue(*node.tensor); pc++; break;
            case 1: desc.kind = ArgDesc::LITERAL; desc.literal = c10::IValue(node.int_val); pc++; break;
            case 2: desc.kind = ArgDesc::LITERAL; desc.literal = c10::IValue(node.float_val); pc++; break;
            case 3: desc.kind = ArgDesc::LITERAL; desc.literal = c10::IValue(node.bool_val); pc++; break;
            case 4: desc.kind = ArgDesc::LITERAL; desc.literal = c10::IValue(std::string(node.string_val)); pc++; break;
            case 5: desc.kind = ArgDesc::LITERAL; desc.literal = c10::IValue(); pc++; break;
            case 9: {
                std::string dev(node.string_val);
                desc.kind = ArgDesc::LITERAL;
                desc.literal = (node.int_val >= 0)
                    ? c10::IValue(c10::Device(dev + ":" + std::to_string(node.int_val)))
                    : c10::IValue(c10::Device(dev));
                pc++;
                break;
            }
            case 7: { // list
                int64_t count = node.child_count;
                pc++;

                if (count == 0) { desc.kind = ArgDesc::INT_LIST_LITERAL; break; }

                int64_t first_tag = graph[pc].tag;
                bool homo = true;
                for (int64_t j = 1; j < count && homo; j++)
                    if (graph[pc + j].tag != first_tag) homo = false;

                if (homo && first_tag == 10) {
                    desc.kind = ArgDesc::TENSOR_LIST_SLOTS;
                    for (int64_t j = 0; j < count; j++) {
                        std::string ref(graph[pc].string_val);
                        desc.slot_list.push_back(name_to_slot.at(ref));
                        pc++;
                    }
                } else if (homo && first_tag == 1) {
                    desc.kind = ArgDesc::INT_LIST_LITERAL;
                    for (int64_t j = 0; j < count; j++) { desc.int_list.push_back(graph[pc].int_val); pc++; }
                } else if (homo && first_tag == 2) {
                    desc.kind = ArgDesc::FLOAT_LIST_LITERAL;
                    for (int64_t j = 0; j < count; j++) { desc.float_list.push_back(graph[pc].float_val); pc++; }
                } else if (homo && first_tag == 3) {
                    desc.kind = ArgDesc::BOOL_LIST_LITERAL;
                    for (int64_t j = 0; j < count; j++) { desc.bool_list.push_back(graph[pc].bool_val); pc++; }
                } else {
                    // Fallback: GenericList literal
                    auto glist = c10::impl::GenericList(c10::AnyType::get());
                    for (int64_t j = 0; j < count; j++) { glist.emplace_back(graph[pc].int_val); pc++; }
                    desc.kind = ArgDesc::LITERAL;
                    desc.literal = c10::IValue(std::move(glist));
                }
                break;
            }
            default:
                throw std::runtime_error("compile_graph: unknown arg tag " + std::to_string(node.tag));
            }
            named_descs.push_back({aname, std::move(desc)});
        }

        // Reorder args to match schema parameter positions at compile time.
        // This handles cases where the export graph omits optional args
        // (e.g., layout in zeros.default) that sit between other args.
        const auto &schema_params = handle.schema().arguments();
        std::vector<ArgDesc> arg_descs;
        arg_descs.reserve(schema_params.size());

        size_t ni = 0;
        for (size_t si = 0; si < schema_params.size(); si++) {
            const auto &param = schema_params[si];
            bool found = false;

            if (ni < named_descs.size()) {
                if (named_descs[ni].first.empty() ||
                    named_descs[ni].first == param.name()) {
                    arg_descs.push_back(std::move(named_descs[ni].second));
                    ni++;
                    found = true;
                } else {
                    // Search by name in remaining args
                    for (size_t nj = ni; nj < named_descs.size(); nj++) {
                        if (named_descs[nj].first == param.name()) {
                            arg_descs.push_back(std::move(named_descs[nj].second));
                            named_descs.erase(named_descs.begin() + nj);
                            found = true;
                            break;
                        }
                    }
                }
            }

            if (!found && param.default_value().has_value()) {
                // Insert schema default as a literal
                ArgDesc def_desc;
                def_desc.kind = ArgDesc::LITERAL;
                def_desc.literal = param.default_value().value();
                arg_descs.push_back(std::move(def_desc));
            }
        }

        compiled->ops.push_back(CompiledOp{
            handle, std::move(arg_descs), std::move(out_slots),
            handle.schema().arguments().size()
        });
    }

    for (const auto &name : output_names) {
        auto it = name_to_slot.find(std::string(name));
        if (it != name_to_slot.end())
            compiled->output_slots.push_back(it->second);
    }
    compiled->num_slots = next_slot;
    return compiled;
}

TensorList run_compiled_graph(
    const std::shared_ptr<CrossCompiledGraph> &compiled,
    TensorList tensors)
{
    auto input_tensors = unpack_tensor_list(std::move(tensors));
    auto output_tensors = compiled->run(std::move(input_tensors));
    return pack_tensor_list(output_tensors);
}
