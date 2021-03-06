#include "extorch_native/src/native.rs.h"
#include "extorch_native/include/utils.h"


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

template<class T>
std::vector<T> unpack_torch_scalar_list(std::vector<torch::Scalar> list) {
    std::vector<T> scalar_typed_list;
    for(torch::Scalar scalar : list) {
        T typed_scalar = scalar.to<T>();
        scalar_typed_list.push_back(typed_scalar);
    }
    return scalar_typed_list;
}

torch::detail::TensorDataContainer get_scalar_list(rust::Vec<Scalar> list) {
    if(list.size() > 0) {
        std::vector<torch::Scalar> scalar_vec;
        Scalar scalar = list[0];
        auto r_entry_used = scalar.entry_used;
        std::string entry_used(r_entry_used.data(), r_entry_used.size());

        for(auto elem: list) {
            torch::Scalar scalar = get_scalar_type(elem);
            scalar_vec.push_back(scalar);
        }

        if(entry_used == "uint8") {
            // torch::ArrayRef<uint8_t> ui8_list = unpack_torch_scalar_list<uint8_t>(scalar_vec);
            std::vector<uint8_t> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                uint8_t typed_scalar = scalar.toByte();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        } else if(entry_used == "int8") {
            std::vector<int8_t> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                int8_t typed_scalar = scalar.toChar();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        } else if (entry_used == "int16") {
            std::vector<int16_t> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                int16_t typed_scalar = scalar.toShort();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        } else if (entry_used == "int32") {
            std::vector<int> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                int typed_scalar = scalar.toInt();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        } else if (entry_used == "int64") {
            std::vector<int64_t> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                int64_t typed_scalar = scalar.toLong();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        } else if (entry_used == "float16") {
            std::vector<float> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                float typed_scalar = scalar.toFloat();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        } else if (entry_used == "float32") {
            std::vector<float> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                float typed_scalar = scalar.toFloat();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        } else if (entry_used == "float64") {
            std::vector<double> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                double typed_scalar = scalar.toDouble();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        } else if (entry_used == "bool") {
            std::vector<uint8_t> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                uint8_t typed_scalar = scalar.toChar();
                scalar_typed_list.push_back(typed_scalar);
            }
            return scalar_typed_list;
        }
    }
    return torch::detail::TensorDataContainer();
}
