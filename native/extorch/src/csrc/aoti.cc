#include "extorch/src/native.rs.h"
#include "extorch/include/aoti.h"

bool aoti_is_available() {
    return EXTORCH_AOTI_AVAILABLE;
}

#if EXTORCH_AOTI_AVAILABLE

std::shared_ptr<CrossAOTILoader> aoti_load(
    rust::String path,
    rust::String model_name,
    int64_t device_index)
{
    std::string path_str(path);
    std::string name_str(model_name);

    auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        path_str, name_str, false, 1, static_cast<c10::DeviceIndex>(device_index));

    return std::make_shared<CrossAOTILoader>(std::move(loader));
}

TensorList aoti_forward(
    const std::shared_ptr<CrossAOTILoader> &loader,
    TensorList inputs)
{
    auto input_tensors = unpack_tensor_list(inputs);
    auto outputs = loader->loader->run(input_tensors);
    return pack_tensor_list(outputs);
}

rust::Vec<rust::String> aoti_get_metadata_keys(
    const std::shared_ptr<CrossAOTILoader> &loader)
{
    rust::Vec<rust::String> result;
    auto metadata = loader->loader->get_metadata();
    for (const auto &kv : metadata) {
        result.push_back(rust::String(kv.first));
    }
    return result;
}

rust::String aoti_get_metadata_value(
    const std::shared_ptr<CrossAOTILoader> &loader,
    rust::String key)
{
    std::string key_str(key);
    auto metadata = loader->loader->get_metadata();
    auto it = metadata.find(key_str);
    if (it == metadata.end()) {
        throw std::runtime_error("Metadata key not found: " + key_str);
    }
    return rust::String(it->second);
}

rust::Vec<rust::String> aoti_get_constant_fqns(
    const std::shared_ptr<CrossAOTILoader> &loader)
{
    rust::Vec<rust::String> result;
    auto fqns = loader->loader->get_constant_fqns();
    for (const auto &name : fqns) {
        result.push_back(rust::String(name));
    }
    return result;
}

#else

std::shared_ptr<CrossAOTILoader> aoti_load(rust::String, rust::String, int64_t) {
    throw std::runtime_error("AOTI support is not available in this libtorch build");
}
TensorList aoti_forward(const std::shared_ptr<CrossAOTILoader>&, TensorList) {
    throw std::runtime_error("AOTI support is not available");
}
rust::Vec<rust::String> aoti_get_metadata_keys(const std::shared_ptr<CrossAOTILoader>&) {
    throw std::runtime_error("AOTI support is not available");
}
rust::String aoti_get_metadata_value(const std::shared_ptr<CrossAOTILoader>&, rust::String) {
    throw std::runtime_error("AOTI support is not available");
}
rust::Vec<rust::String> aoti_get_constant_fqns(const std::shared_ptr<CrossAOTILoader>&) {
    throw std::runtime_error("AOTI support is not available");
}

#endif
