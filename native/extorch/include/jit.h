#pragma once
#include "common.h"
#include "utils.h"
#include <torch/script.h>

struct CrossModuleImpl {
    torch::jit::script::Module module;
    CrossModuleImpl(torch::jit::script::Module m) : module(std::move(m)) {}
};

struct IValueNode;
struct IValueFlat;
struct NamedTensor;

// Load/save
std::shared_ptr<CrossModule> jit_load(
    rust::String path,
    struct Device s_device);

void jit_save(
    const std::shared_ptr<CrossModule> &module,
    rust::String path);

// Forward pass / method invocation
IValueFlat jit_forward(
    const std::shared_ptr<CrossModule> &module,
    TensorList inputs);

IValueFlat jit_invoke_method(
    const std::shared_ptr<CrossModule> &module,
    rust::String method_name,
    TensorList inputs);

// Module introspection
rust::Vec<rust::String> jit_get_method_names(
    const std::shared_ptr<CrossModule> &module);

rust::Vec<NamedTensor> jit_named_parameters(
    const std::shared_ptr<CrossModule> &module);

rust::Vec<NamedTensor> jit_named_buffers(
    const std::shared_ptr<CrossModule> &module);

rust::Vec<rust::String> jit_named_modules(
    const std::shared_ptr<CrossModule> &module);

// Mode setting
void jit_set_eval(
    const std::shared_ptr<CrossModule> &module);

void jit_set_train(
    const std::shared_ptr<CrossModule> &module);

// Device transfer
std::shared_ptr<CrossModule> jit_to_device(
    const std::shared_ptr<CrossModule> &module,
    struct Device s_device);

// IValue flattening helper
IValueFlat flatten_ivalue(const torch::jit::IValue &ivalue);

// IR introspection
struct ParameterInfo;
struct SubmoduleInfo;
struct ModuleInfo;

rust::String jit_graph_str(
    const std::shared_ptr<CrossModule> &module);

rust::Vec<ParameterInfo> jit_module_parameters_info(
    const std::shared_ptr<CrossModule> &module);

rust::Vec<SubmoduleInfo> jit_module_submodules_info(
    const std::shared_ptr<CrossModule> &module);

rust::Vec<rust::String> jit_module_methods_info(
    const std::shared_ptr<CrossModule> &module);

rust::Vec<SubmoduleInfo> jit_all_submodules_info(
    const std::shared_ptr<CrossModule> &module);
