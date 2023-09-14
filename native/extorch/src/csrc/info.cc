#include "extorch/src/native.rs.h"
#include "extorch/include/info.h"
#include "extorch/include/printing.h"


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
    auto type = cross_tensor.dtype().toScalarType();
    auto type_name = inv_type_mapping[type];
    rust::String type_rust(type_name.data(), type_name.size());
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

rust::String repr(const std::shared_ptr<CrossTensor> &tensor, const PrintOptions opts) {
    CrossTensor cross_tensor = *tensor.get();
    auto str_repr = _tensor_str(cross_tensor, opts, 0);
    // auto str_repr = cross_tensor.toString();
    // std::stringstream ss;
    // ss << cross_tensor;
    // auto str_repr = ss.str();
    rust::String tensor_repr(str_repr.data(), str_repr.size());
    return tensor_repr;
}

template<typename scalar_t>
Scalar pack_complex_scalar(scalar_t scalar, torch::ScalarType type) {
    Scalar scalar_s;
    std::string identifier = "complex32";
    if(type == torch::ScalarType::ComplexFloat) {
        identifier = "complex64";
    } else if (type == torch::ScalarType::ComplexDouble) {
        identifier = "complex128";
    }

    auto re = static_cast<double>(scalar.real());
    auto imag = static_cast<double>(scalar.imag());

    auto re_repr = static_cast<uint8_t*>(static_cast<void*>(&re));
    auto im_repr = static_cast<uint8_t*>(static_cast<void*>(&imag));

    rust::Vec<uint8_t> vec_repr;
    std::copy(re_repr, re_repr + sizeof(double), std::back_inserter(vec_repr));
    std::copy(im_repr, im_repr + sizeof(double), std::back_inserter(vec_repr));

    rust::string entry_used(identifier.data(), identifier.size());
    scalar_s.entry_used = entry_used;
    scalar_s._repr = vec_repr;
    return scalar_s;
}

template<typename scalar_t>
Scalar pack_scalar(scalar_t scalar, torch::ScalarType type) {
    Scalar scalar_s;
    std::string identifier = "float32";
    if(type == torch::ScalarType::Bool) {
        // scalar_s._bool = (bool) scalar;
        identifier = "bool";
    } else if(type == torch::ScalarType::Byte) {
        // scalar_s._ui8 = scalar;
        identifier = "uint8";
    } else if(type == torch::ScalarType::Char) {
        // scalar_s._i8 = scalar;
        identifier = "int8";
    } else if(type == torch::ScalarType::Short) {
        // scalar_s._i16 = scalar;
        identifier = "int16";
    } else if(type == torch::ScalarType::Int) {
        // scalar_s._i32 = scalar;
        identifier = "int32";
    } else if(type == torch::ScalarType::Long) {
        // scalar_s._i64 = scalar;
        identifier = "int64";
    } else if(type == torch::ScalarType::Half) {
        // scalar_s._f16 = (float) scalar;
        identifier = "float32";
    } else if(type == torch::ScalarType::Float) {
        // scalar_s._f32 = scalar;
        identifier = "float32";
    } else if(type == torch::ScalarType::Double) {
        // scalar_s._f64 = scalar;
        identifier = "float64";
    }
    // } else if(type == torch::ScalarType::ComplexHalf) {
    //     return pack_complex_scalar<c10::Half>(scalar, type);
    // } else if(type == torch::ScalarType::ComplexFloat) {
    //     return pack_complex_scalar<float>(scalar, type);
    // } else if (type == torch::ScalarType::ComplexDouble) {
    //     return pack_complex_scalar<double>(scalar, type);
    // }

    auto scalar_repr = static_cast<uint8_t*>(static_cast<void*>(&scalar));
    rust::Vec<uint8_t> vec_repr;
    std::copy(scalar_repr, scalar_repr + sizeof(scalar_t), std::back_inserter(vec_repr));

    rust::string entry_used(identifier.data(), identifier.size());
    scalar_s.entry_used = entry_used;
    scalar_s._repr = vec_repr;
    return scalar_s;
}

ScalarList to_list(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto conv_tensor = cross_tensor.contiguous();
    auto size = cross_tensor.sizes().vec();

    rust::Vec<int64_t> rust_size;
    for(auto dim: size) {
        rust_size.push_back(dim);
    }

    auto numel = cross_tensor.numel();

    rust::Vec<Scalar> scalar_list;

    AT_DISPATCH_SWITCH(cross_tensor.scalar_type(), "to_list",
        AT_DISPATCH_CASE_ALL_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::Bool,
            [&] {
                auto data_ptr = conv_tensor.data_ptr<scalar_t>();
                for(int64_t i = 0; i < numel; i++) {
                    auto scalar = pack_scalar<scalar_t>(data_ptr[i], cross_tensor.scalar_type());
                    scalar_list.push_back(scalar);
                }
        })
        AT_DISPATCH_CASE_COMPLEX_TYPES_AND(torch::ScalarType::ComplexHalf,
            [&] {
                auto data_ptr = conv_tensor.data_ptr<scalar_t>();
                for(int64_t i = 0; i < numel; i++) {
                    auto scalar = pack_complex_scalar<scalar_t>(data_ptr[i], cross_tensor.scalar_type());
                    scalar_list.push_back(scalar);
                }
        })
    );

    ScalarList result {
        std::move(scalar_list),
        std::move(rust_size)
    };
    // result.list = ;
    // result.size = rust_size;
    return result;
}

Scalar item(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto scalar_value = cross_tensor.item();
    Scalar scalar;

    AT_DISPATCH_SWITCH(cross_tensor.scalar_type(), "to_list",
        AT_DISPATCH_CASE_ALL_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::Bool,
        [&] {
            scalar = pack_scalar<scalar_t>(scalar_value.to<scalar_t>(), cross_tensor.scalar_type());
        })
        AT_DISPATCH_CASE_COMPLEX_TYPES_AND(torch::ScalarType::ComplexHalf,
        [&] {
            scalar = pack_complex_scalar<scalar_t>(scalar_value.to<scalar_t>(), cross_tensor.scalar_type());
        })
    );

    return std::move(scalar);
}

bool requires_grad(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.requires_grad();
}

int64_t numel(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.numel();
}

rust::String memory_format(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto type = cross_tensor.suggest_memory_format();
    auto type_name = inv_memory_fmt_mapping[type];
    rust::String type_rust(type_name.data(), type_name.size());
    return type_rust;
}

rust::String layout(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto type = cross_tensor.layout();
    auto type_name = inv_layout_mapping[type];
    rust::String type_rust(type_name.data(), type_name.size());
    return type_rust;
}

bool is_complex(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_complex();
}

bool is_floating_point(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_floating_point();
}

bool is_conj(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_conj();
}

bool is_nonzero(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_nonzero();
}
