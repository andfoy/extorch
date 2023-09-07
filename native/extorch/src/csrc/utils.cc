#include "extorch/src/native.rs.h"
#include "extorch/include/utils.h"
#include <complex>


std::unordered_map<std::string, torch::ScalarType> type_mapping = {
    {"int", torch::kInt},
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
    {"complex_half", torch::kComplexHalf},
    {"complex_float", torch::kComplexFloat},
    {"complex_double", torch::kComplexDouble},
    {"complex32", torch::kComplexHalf},
    {"complex64", torch::kComplexFloat},
    {"complex128", torch::kComplexDouble},
    {"bool", torch::kBool}};

std::unordered_map<torch::ScalarType, std::string> inv_type_mapping = {
    {torch::kBool, "bool"},
    {torch::kByte, "byte"},
    {torch::kShort, "short"},
    {torch::kInt, "int"},
    {torch::kLong, "long"},
    {torch::kShort, "short"},
    {torch::kHalf, "half"},
    {torch::kFloat, "float"},
    {torch::kDouble, "double"},
    {torch::kComplexHalf, "complex_half"},
    {torch::kComplexFloat, "complex_float"},
    {torch::kComplexDouble, "complex_double"},
};

std::unordered_map<std::string, torch::DeviceType> device_mapping = {
    {"cpu", torch::kCPU},
    {"cuda", torch::kCUDA},
    {"hip", torch::kHIP},
    {"fpga", torch::kFPGA},
    {"vulkan", torch::kVulkan},
    {"xla", torch::kXLA}};

std::unordered_map<std::string, torch::Layout> layout_mapping = {
    {"strided", torch::kStrided},
    {"sparse", torch::kSparse}};

std::unordered_map<torch::Layout, std::string> inv_layout_mapping = {
    {torch::kStrided, "strided"},
    {torch::kSparse, "sparse"}
};

std::unordered_map<std::string, torch::MemoryFormat> memory_fmt_mapping = {
    {"contiguous", torch::MemoryFormat::Contiguous},
    {"preserve", torch::MemoryFormat::Preserve},
    {"channels_last", torch::MemoryFormat::ChannelsLast},
    {"channels_last_3d", torch::MemoryFormat::ChannelsLast3d}};

std::unordered_map<torch::MemoryFormat, std::string> inv_memory_fmt_mapping = {
    {torch::MemoryFormat::Contiguous, "contiguous"},
    {torch::MemoryFormat::Preserve, "preserve"},
    {torch::MemoryFormat::ChannelsLast, "channels_last"},
    {torch::MemoryFormat::ChannelsLast3d, "channels_last_3d"}
};

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

template<typename T>
T copy_bytes_to_type(const uint8_t* bytes) {
    T recov;
    std::copy(reinterpret_cast<const uint8_t*>(bytes),
              reinterpret_cast<const uint8_t*>(bytes + sizeof(T)),
              reinterpret_cast<uint8_t*>(&recov));
    return recov;
}

template<typename T>
c10::complex<T> copy_complex_bytes_to_type(const uint8_t* bytes) {
    T real;
    T im;
    real = copy_bytes_to_type<double>(bytes);
    im = copy_bytes_to_type<double>(bytes + sizeof(double));
    return c10::complex<T>(real, im);
}

template<typename T>
c10::complex<T> get_complex_type(Scalar scalar) {
    auto r_entry_used = scalar.entry_used;
    std::string entry_used(r_entry_used.data(), r_entry_used.size());

    c10::complex<T> ret_scalar;
    uint8_t* scalar_ptr = scalar._repr.data();

    if (entry_used == "complex32") {
        ret_scalar = copy_complex_bytes_to_type<T>(scalar_ptr);
    } else if (entry_used == "complex64") {
        ret_scalar = copy_complex_bytes_to_type<T>(scalar_ptr);
    } else if (entry_used == "complex128") {
        ret_scalar = copy_complex_bytes_to_type<T>(scalar_ptr);
    }
    return ret_scalar;
}

torch::Scalar get_scalar_type(Scalar scalar) {
    auto r_entry_used = scalar.entry_used;
    std::string entry_used(r_entry_used.data(), r_entry_used.size());

    torch::Scalar ret_scalar;
    uint8_t* scalar_ptr = scalar._repr.data();
    if(entry_used == "uint8") {
        ret_scalar = copy_bytes_to_type<uint8_t>(scalar_ptr);
    } else if(entry_used == "int8") {
        ret_scalar = copy_bytes_to_type<int8_t>(scalar_ptr);
    } else if (entry_used == "int16") {
        ret_scalar = copy_bytes_to_type<int16_t>(scalar_ptr);
    } else if (entry_used == "int32") {
        ret_scalar = copy_bytes_to_type<int32_t>(scalar_ptr);
    } else if (entry_used == "int64") {
        ret_scalar = copy_bytes_to_type<int64_t>(scalar_ptr);
    } else if (entry_used == "float16") {
        ret_scalar = copy_bytes_to_type<float>(scalar_ptr);
    } else if (entry_used == "float32") {
        ret_scalar = copy_bytes_to_type<float>(scalar_ptr);
    } else if (entry_used == "float64") {
        ret_scalar = copy_bytes_to_type<double>(scalar_ptr);
    } else if (entry_used == "complex32") {
        ret_scalar = copy_bytes_to_type<c10::complex<double>>(scalar_ptr);
    } else if (entry_used == "complex64") {
        ret_scalar = copy_bytes_to_type<c10::complex<double>>(scalar_ptr);
    } else if (entry_used == "complex128") {
        ret_scalar = copy_bytes_to_type<c10::complex<double>>(scalar_ptr);
    } else if (entry_used == "bool") {
        ret_scalar = scalar_ptr[0];
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

        if(entry_used.rfind("complex", 0) != 0) {
            for(auto elem: list) {
                torch::Scalar scalar = get_scalar_type(elem);
                scalar_vec.push_back(scalar);
            }
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
            std::vector<c10::Half> scalar_typed_list;
            for(torch::Scalar scalar : scalar_vec) {
                float typed_scalar = scalar.toHalf();
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

torch::detail::TensorDataContainer get_complex_tensor_parts(
        rust::Vec<Scalar> list,
        c10::ScalarType scalar_type,
        torch::TensorOptions opts,
        const int64_t *ptr) {

    if(scalar_type == torch::kComplexHalf || scalar_type == torch::kComplexFloat) {
        std::vector<c10::complex<float>> typed_list;
        // std::vector<float> im_typed_list;
        for(auto elem: list) {
            auto cmplx = get_complex_type<float>(elem);
            typed_list.push_back(cmplx);
            // im_typed_list.push_back(cmplx.imag());
        }
        return typed_list;
    } else {
        std::vector<c10::complex<double>> typed_list;
        for(auto elem: list) {
            auto cmplx = get_complex_type<double>(elem);
            typed_list.push_back(cmplx);
        }
        return typed_list;
    }
}