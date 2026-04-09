#pragma once
#include "common.h"
#include "utils.h"

#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION) && !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#define EXTORCH_AOTI_AVAILABLE 1
#else
#define EXTORCH_AOTI_AVAILABLE 0
#endif

// Always define the struct so cxx bridge can reference it.
// The loader pointer is null when AOTI is not available.
struct CrossAOTILoaderImpl {
#if EXTORCH_AOTI_AVAILABLE
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
    CrossAOTILoaderImpl(std::unique_ptr<torch::inductor::AOTIModelPackageLoader> l)
        : loader(std::move(l)) {}
#else
    CrossAOTILoaderImpl() {}
#endif
};

std::shared_ptr<CrossAOTILoader> aoti_load(
    rust::String path,
    rust::String model_name,
    int64_t device_index);

TensorList aoti_forward(
    const std::shared_ptr<CrossAOTILoader> &loader,
    TensorList inputs);

rust::Vec<rust::String> aoti_get_metadata_keys(
    const std::shared_ptr<CrossAOTILoader> &loader);

rust::String aoti_get_metadata_value(
    const std::shared_ptr<CrossAOTILoader> &loader,
    rust::String key);

rust::Vec<rust::String> aoti_get_constant_fqns(
    const std::shared_ptr<CrossAOTILoader> &loader);

bool aoti_is_available();
