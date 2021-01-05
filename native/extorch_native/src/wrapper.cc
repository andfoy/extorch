#include "extorch_native/include/wrapper.h"
#include "extorch_native/src/lib.rs.h"
#include <iostream>
// #include <string>

std::unordered_map<std::string, torch::ScalarType> type_mapping = {
    {"int", torch::kI32},
    {"int8", torch::kI8},
    {"int16", torch::kI16},
    {"int32", torch::kI32},
    {"int64", torch::kI64},
    {"uint8", torch::kUInt8},
    {"float", torch::kF32},
    {"float16", torch::kF16},
    {"float32", torch::kF32},
    {"float64", torch::kF64},
    {"bfloat16", torch::kBFloat16},
    {"byte", torch::kByte},
    {"half", torch::kHalf},
    {"char", torch::kChar},
    {"short", torch::kShort},
    {"double", torch::kDouble},
    {"long", torch::kLong},
    {"complex32", torch::kComplexHalf},
    {"complex64", torch::kComplexFloat},
    {"complex128", torch::kComplexDouble},
    {"bool", torch::kBool}};

std::unordered_map<std::string, torch::DeviceType> device_mapping = {
    {"cpu", torch::kCPU},
    {"cuda", torch::kCUDA},
    {"hip", torch::kHIP},
    {"fpga", torch::kFPGA},
    {"vulkan", torch::kVulkan},
    {"msnpu", torch::kMSNPU},
    {"xla", torch::kXLA}};

std::unordered_map<std::string, torch::Layout> layout_mapping = {
    {"strided", torch::kStrided},
    {"sparse", torch::kSparse}};

std::unordered_map<std::string, torch::MemoryFormat> memory_fmt_mapping = {
    {"contiguous", torch::MemoryFormat::Contiguous},
    {"preserve", torch::MemoryFormat::Preserve},
    {"channels_last", torch::MemoryFormat::ChannelsLast},
    {"channels_last_3d", torch::MemoryFormat::ChannelsLast3d}};

torch::TensorOptions get_tensor_options(rust::String s_dtype,
                                        rust::String s_layout, Device ddevice,
                                        bool requires_grad, bool pin_memory,
                                        rust::String s_mem_fmt)

{
    auto s_device = ddevice.device;
    auto device_index = ddevice.index;

    std::string dtype_str(s_dtype.data(), s_dtype.size());
    std::string layout_str(s_layout.data(), s_layout.size());
    std::string device_str(s_device.data(), s_device.size());
    std::string mem_fmt_str(s_mem_fmt.data(), s_mem_fmt.size());

    auto type = torch::get_default_dtype();
    auto type_search = type_mapping.find(dtype_str);
    if (type_search != type_mapping.end())
    {
        type = torch::scalarTypeToTypeMeta(type_search->second);
    }

    auto device = torch::Device(torch::kCPU);
    auto device_search = device_mapping.find(device_str);
    if (device_search != device_mapping.end()) {
        device = torch::Device(device_search->second, device_index);
    } else {
        device = torch::Device(device_str);
    }

    auto layout = torch::kStrided;
    auto layout_search = layout_mapping.find(layout_str);
    if (layout_search != layout_mapping.end()) {
        layout = layout_search->second;
    }

    auto mem_format = torch::MemoryFormat::Contiguous;
    auto mem_fmt_search = memory_fmt_mapping.find(mem_fmt_str);
    if (mem_fmt_search != memory_fmt_mapping.end()) {
        mem_format = mem_fmt_search->second;
    }

    torch::TensorOptions tensor_options =
        torch::TensorOptions(type).
        device(device).
        layout(layout).
        requires_grad(requires_grad).
        memory_format(mem_format).
        pinned_memory(pin_memory);

    return tensor_options;
}

torch::Scalar get_scalar_type(Scalar scalar) {
    auto r_entry_used = scalar.entry_used;
    std::string entry_used(r_entry_used.data(), r_entry_used.size());

    torch::Scalar ret_scalar;
    if(entry_used == "uint8") {
        ret_scalar = scalar._ui8;
    } else if(entry_used == "int8") {
        ret_scalar = scalar._i8;
    } else if (entry_used == "int16") {
        ret_scalar = scalar._i16;
    } else if (entry_used == "int32") {
        ret_scalar = scalar._i32;
    } else if (entry_used == "int64") {
        ret_scalar = scalar._i64;
    } else if (entry_used == "float16") {
        ret_scalar = scalar._f16;
    } else if (entry_used == "float32") {
        ret_scalar = scalar._f32;
    } else if (entry_used == "float64") {
        ret_scalar = scalar._f64;
    } else if (entry_used == "bool") {
        ret_scalar = scalar._bool;
    }
    return ret_scalar;
}

std::shared_ptr<CrossTensor> empty(
    rust::Vec<int64_t> dims, rust::String s_dtype,
    rust::String s_layout, Device s_device,
    bool requires_grad, bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::empty(torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> zeros(
    rust::Vec<int64_t> dims, rust::String s_dtype,
    rust::String s_layout, Device s_device,
    bool requires_grad, bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::zeros(torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> ones(
    rust::Vec<int64_t> dims, rust::String s_dtype,
    rust::String s_layout, Device s_device,
    bool requires_grad, bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::ones(torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> full(
    rust::Vec<int64_t> dims,
    Scalar scalar,
    rust::String s_dtype,
    rust::String s_layout,
    Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    auto torch_scalar = get_scalar_type(scalar);
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::full(torch::IntArrayRef{ptr, dims.size()}, torch_scalar, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> eye(
    int64_t n,
    int64_t m,
    rust::String s_dtype,
    rust::String s_layout,
    Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    // auto torch_scalar = get_scalar_type(scalar);
    // const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::eye(n, m, opts);
    // torch::Tensor tensor = torch::full(torch::IntArrayRef{ptr, dims.size()}, torch_scalar, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor> &tensor)
{
    CrossTensor cross_tensor = *tensor.get();
    auto sizes = cross_tensor.sizes();
    rust::Slice<const int64_t> slice{sizes.data(), sizes.size()};
    return slice;
}

rust::String dtype(const std::shared_ptr<CrossTensor> &tensor)
{
    CrossTensor cross_tensor = *tensor.get();
    auto type = cross_tensor.dtype();
    auto it = std::find_if(
        type_mapping.begin(), type_mapping.end(),
        [&type](const std::pair<std::string, torch::ScalarType> &p) {
            return p.second == type;
        });
    auto type_name = it->first;
    rust::String type_rust(type_name.data(), type_name.size());
    // rust::Slice<const int64_t> slice{sizes.data(), sizes.size()};
    return type_rust;
}

Device device(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto device = cross_tensor.device();
    auto device_type = device.type();
    auto device_index = device.index();
    auto it = std::find_if(
        device_mapping.begin(), device_mapping.end(),
        [&device_type](const std::pair<std::string, torch::DeviceType> &p) {
            return p.second == device_type;
        });
    auto device_name = it->first;
    rust::String device_rust(device_name.data(), device_name.size());
    Device this_device { device_rust, device_index };
    return this_device;
}

rust::String repr(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    // auto str_repr = cross_tensor.toString();
    std::stringstream ss;
    ss << cross_tensor;
    auto str_repr = ss.str();
    rust::String tensor_repr(str_repr.data(), str_repr.size());
    return tensor_repr;
}
