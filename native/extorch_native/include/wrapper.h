#pragma once
#include "rust/cxx.h"
#include <memory>
// #include <string>
#include <torch/torch.h>

// struct CrossTensorRef;

struct Device;
using CrossTensor = torch::Tensor;
std::shared_ptr<CrossTensor> empty(
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);
rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor> &tensor);
rust::String dtype(const std::shared_ptr<CrossTensor> &tensor);
Device device(const std::shared_ptr<CrossTensor> &tensor);
