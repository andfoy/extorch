#include "extorch/src/native.rs.h"
#include "extorch/include/jit.h"


// Helper: convert Device struct to torch::Device
static torch::Device make_torch_device(const Device &s_device) {
    std::string device_str(s_device.device);
    auto it = device_mapping.find(device_str);
    if (it == device_mapping.end()) {
        throw std::runtime_error("Unknown device: " + device_str);
    }
    if (s_device.index >= 0) {
        return torch::Device(it->second, s_device.index);
    }
    return torch::Device(it->second);
}

// Helper: recursively flatten an IValue into a pre-order node list
static void flatten_ivalue_recursive(
    const torch::jit::IValue &ivalue,
    rust::Vec<IValueNode> &nodes,
    int64_t parent_idx)
{
    IValueNode node;
    node.parent_idx = parent_idx;
    node.child_count = 0;
    node.int_val = 0;
    node.float_val = 0.0;
    node.bool_val = false;
    node.string_val = rust::String("");
    // Initialize tensor to nullptr - cxx SharedPtr default
    node.tensor = std::shared_ptr<CrossTensor>(nullptr);

    if (ivalue.isTensor()) {
        node.tag = 0;
        node.tensor = std::make_shared<CrossTensor>(ivalue.toTensor());
        nodes.push_back(std::move(node));
    } else if (ivalue.isInt()) {
        node.tag = 1;
        node.int_val = ivalue.toInt();
        nodes.push_back(std::move(node));
    } else if (ivalue.isDouble()) {
        node.tag = 2;
        node.float_val = ivalue.toDouble();
        nodes.push_back(std::move(node));
    } else if (ivalue.isBool()) {
        node.tag = 3;
        node.bool_val = ivalue.toBool();
        nodes.push_back(std::move(node));
    } else if (ivalue.isString()) {
        node.tag = 4;
        node.string_val = rust::String(ivalue.toStringRef());
        nodes.push_back(std::move(node));
    } else if (ivalue.isNone()) {
        node.tag = 5;
        nodes.push_back(std::move(node));
    } else if (ivalue.isTuple()) {
        node.tag = 6;
        auto tuple = ivalue.toTuple();
        node.child_count = static_cast<int64_t>(tuple->elements().size());
        int64_t my_idx = static_cast<int64_t>(nodes.size());
        nodes.push_back(std::move(node));
        for (const auto &elem : tuple->elements()) {
            flatten_ivalue_recursive(elem, nodes, my_idx);
        }
    } else if (ivalue.isList()) {
        node.tag = 7;
        auto list = ivalue.toList();
        node.child_count = static_cast<int64_t>(list.size());
        int64_t my_idx = static_cast<int64_t>(nodes.size());
        nodes.push_back(std::move(node));
        for (const auto &elem : list) {
            flatten_ivalue_recursive(elem, nodes, my_idx);
        }
    } else if (ivalue.isGenericDict()) {
        node.tag = 8;
        auto dict = ivalue.toGenericDict();
        node.child_count = static_cast<int64_t>(dict.size()) * 2; // key-value pairs
        int64_t my_idx = static_cast<int64_t>(nodes.size());
        nodes.push_back(std::move(node));
        for (const auto &entry : dict) {
            flatten_ivalue_recursive(entry.key(), nodes, my_idx);
            flatten_ivalue_recursive(entry.value(), nodes, my_idx);
        }
    } else {
        // Fallback: represent as string
        node.tag = 4;
        std::ostringstream oss;
        oss << ivalue;
        node.string_val = rust::String(oss.str());
        nodes.push_back(std::move(node));
    }
}

IValueFlat flatten_ivalue(const torch::jit::IValue &ivalue) {
    IValueFlat result;
    flatten_ivalue_recursive(ivalue, result.nodes, -1);
    return result;
}

// Helper: convert TensorList to IValue vector
static std::vector<torch::jit::IValue> make_inputs(TensorList inputs)
{
    auto tensors = unpack_tensor_list(inputs);
    std::vector<torch::jit::IValue> ivalue_inputs;
    ivalue_inputs.reserve(tensors.size());
    for (auto &tensor : tensors) {
        ivalue_inputs.emplace_back(tensor);
    }
    return ivalue_inputs;
}

// ============================================================================
// Load / Save
// ============================================================================

std::shared_ptr<CrossModule> jit_load(
    rust::String path,
    Device s_device)
{
    std::string path_str(path);
    auto device = make_torch_device(s_device);
    auto module = torch::jit::load(path_str, device);
    return std::make_shared<CrossModule>(std::move(module));
}

void jit_save(
    const std::shared_ptr<CrossModule> &module,
    rust::String path)
{
    std::string path_str(path);
    module->module.save(path_str);
}

// ============================================================================
// Forward / Invoke
// ============================================================================

IValueFlat jit_forward(
    const std::shared_ptr<CrossModule> &module,
    TensorList inputs)
{
    auto ivalue_inputs = make_inputs(std::move(inputs));
    auto result = module->module.forward(ivalue_inputs);
    return flatten_ivalue(result);
}

IValueFlat jit_invoke_method(
    const std::shared_ptr<CrossModule> &module,
    rust::String method_name,
    TensorList inputs)
{
    std::string name_str(method_name);
    auto ivalue_inputs = make_inputs(std::move(inputs));
    auto method = module->module.get_method(name_str);
    auto result = method(ivalue_inputs);
    return flatten_ivalue(result);
}

// ============================================================================
// Introspection
// ============================================================================

rust::Vec<rust::String> jit_get_method_names(
    const std::shared_ptr<CrossModule> &module)
{
    rust::Vec<rust::String> names;
    for (const auto &method : module->module.get_methods()) {
        names.push_back(rust::String(method.name()));
    }
    return names;
}

rust::Vec<NamedTensor> jit_named_parameters(
    const std::shared_ptr<CrossModule> &module)
{
    rust::Vec<NamedTensor> params;
    for (const auto &param : module->module.named_parameters()) {
        NamedTensor nt;
        nt.name = rust::String(param.name);
        nt.tensor = std::make_shared<CrossTensor>(param.value);
        params.push_back(std::move(nt));
    }
    return params;
}

rust::Vec<NamedTensor> jit_named_buffers(
    const std::shared_ptr<CrossModule> &module)
{
    rust::Vec<NamedTensor> buffers;
    for (const auto &buf : module->module.named_buffers()) {
        NamedTensor nt;
        nt.name = rust::String(buf.name);
        nt.tensor = std::make_shared<CrossTensor>(buf.value);
        buffers.push_back(std::move(nt));
    }
    return buffers;
}

rust::Vec<rust::String> jit_named_modules(
    const std::shared_ptr<CrossModule> &module)
{
    rust::Vec<rust::String> names;
    for (const auto &submod : module->module.named_modules()) {
        // Skip the root module (empty name)
        if (!submod.name.empty()) {
            names.push_back(rust::String(submod.name));
        }
    }
    return names;
}

// ============================================================================
// Mode setting
// ============================================================================

void jit_set_eval(const std::shared_ptr<CrossModule> &module) {
    module->module.eval();
}

void jit_set_train(const std::shared_ptr<CrossModule> &module) {
    module->module.train();
}

// ============================================================================
// Device transfer
// ============================================================================

std::shared_ptr<CrossModule> jit_to_device(
    const std::shared_ptr<CrossModule> &module,
    Device s_device)
{
    auto device = make_torch_device(s_device);
    auto cloned = module->module.clone();
    cloned.to(device);
    return std::make_shared<CrossModule>(std::move(cloned));
}

// ============================================================================
// IR Introspection
// ============================================================================

rust::String jit_graph_str(const std::shared_ptr<CrossModule> &module)
{
    auto method = module->module.get_method("forward");
    auto graph = method.graph();
    std::ostringstream oss;
    oss << *graph;
    return rust::String(oss.str());
}

rust::Vec<ParameterInfo> jit_module_parameters_info(
    const std::shared_ptr<CrossModule> &module)
{
    rust::Vec<ParameterInfo> result;
    for (const auto &param : module->module.named_parameters()) {
        ParameterInfo info;
        info.name = rust::String(param.name);

        auto tensor = param.value;
        for (auto s : tensor.sizes()) {
            info.shape.push_back(s);
        }
        info.dtype = rust::String(inv_type_mapping[tensor.dtype().toScalarType()]);
        info.requires_grad = tensor.requires_grad();

        result.push_back(std::move(info));
    }
    return result;
}

rust::Vec<SubmoduleInfo> jit_module_submodules_info(
    const std::shared_ptr<CrossModule> &module)
{
    rust::Vec<SubmoduleInfo> result;
    for (const auto &submod : module->module.named_children()) {
        SubmoduleInfo info;
        info.name = rust::String(submod.name);
        info.type_name = rust::String(submod.value.type()->name().value_or(c10::QualifiedName("Unknown")).qualifiedName());

        // Get parameter info for this submodule
        for (const auto &param : submod.value.named_parameters(false)) {
            ParameterInfo pinfo;
            pinfo.name = rust::String(param.name);
            auto tensor = param.value;
            for (auto s : tensor.sizes()) {
                pinfo.shape.push_back(s);
            }
            pinfo.dtype = rust::String(inv_type_mapping[tensor.dtype().toScalarType()]);
            pinfo.requires_grad = tensor.requires_grad();
            info.parameters.push_back(std::move(pinfo));
        }

        result.push_back(std::move(info));
    }
    return result;
}

rust::Vec<SubmoduleInfo> jit_all_submodules_info(
    const std::shared_ptr<CrossModule> &module)
{
    rust::Vec<SubmoduleInfo> result;
    // named_modules() returns ALL modules recursively with dotted paths
    for (const auto &submod : module->module.named_modules()) {
        // Skip the root module (empty name)
        if (submod.name.empty()) continue;

        // Check if this is a leaf module (has no children)
        bool has_children = false;
        for (const auto &_ : submod.value.named_children()) {
            has_children = true;
            (void)_;
            break;
        }

        SubmoduleInfo info;
        info.name = rust::String(submod.name);
        info.type_name = rust::String(
            submod.value.type()->name().value_or(
                c10::QualifiedName("Unknown")).qualifiedName());

        // Get parameter info for this specific module (non-recursive)
        for (const auto &param : submod.value.named_parameters(false)) {
            ParameterInfo pinfo;
            pinfo.name = rust::String(param.name);
            auto tensor = param.value;
            for (auto s : tensor.sizes()) {
                pinfo.shape.push_back(s);
            }
            pinfo.dtype = rust::String(inv_type_mapping[tensor.dtype().toScalarType()]);
            pinfo.requires_grad = tensor.requires_grad();
            info.parameters.push_back(std::move(pinfo));
        }

        result.push_back(std::move(info));
    }
    return result;
}

rust::Vec<rust::String> jit_module_methods_info(
    const std::shared_ptr<CrossModule> &module)
{
    rust::Vec<rust::String> result;
    for (const auto &method : module->module.get_methods()) {
        result.push_back(rust::String(method.name()));
    }
    return result;
}
