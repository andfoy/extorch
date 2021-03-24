#include "extorch_native/src/lib.rs.h"
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
