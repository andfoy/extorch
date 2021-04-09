#include "extorch_native/src/native.rs.h"
#include "extorch_native/include/info.h"


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


template<typename scalar_t>
Scalar pack_scalar(scalar_t scalar, torch::ScalarType type) {
    Scalar scalar_s;
    std::string identifier = "float32";
    if(type == torch::ScalarType::Bool) {
        scalar_s._bool = (bool) scalar;
        identifier = "bool";
    } else if(type == torch::ScalarType::Byte) {
        scalar_s._ui8 = scalar;
        identifier = "uint8";
    } else if(type == torch::ScalarType::Char) {
        scalar_s._i8 = scalar;
        identifier = "int8";
    } else if(type == torch::ScalarType::Short) {
        scalar_s._i16 = scalar;
        identifier = "int16";
    } else if(type == torch::ScalarType::Int) {
        scalar_s._i32 = scalar;
        identifier = "int32";
    } else if(type == torch::ScalarType::Long) {
        scalar_s._i64 = scalar;
        identifier = "int64";
    } else if(type == torch::ScalarType::Half) {
        scalar_s._f16 = (float) scalar;
        identifier = "float32";
    } else if(type == torch::ScalarType::Float) {
        scalar_s._f32 = scalar;
        identifier = "float32";
    } else if(type == torch::ScalarType::Double) {
        scalar_s._f64 = scalar;
        identifier = "float64";
    } else if(type == torch::ScalarType::Double) {
        scalar_s._f64 = scalar;
        identifier = "float64";
    }

    rust::string entry_used(identifier.data(), identifier.size());
    scalar_s.entry_used = entry_used;
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
    AT_DISPATCH_ALL_TYPES_AND(torch::ScalarType::Half,
        cross_tensor.scalar_type(), "to_list", [&] {
            auto data_ptr = conv_tensor.data_ptr<scalar_t>();
            for(int64_t i = 0; i < numel; i++) {
                auto scalar = pack_scalar<scalar_t>(data_ptr[i], cross_tensor.scalar_type());
                scalar_list.push_back(scalar);
            }
    });

    ScalarList result {
        std::move(scalar_list),
        std::move(rust_size)
    };
    // result.list = ;
    // result.size = rust_size;
    return result;
}