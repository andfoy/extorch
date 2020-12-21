#pragma once
#include "rust/cxx.h"
#include <memory>
// #include <string>
#include <torch/torch.h>


// struct CrossTensorRef;

using CrossTensor = torch::Tensor;
std::shared_ptr<CrossTensor> empty(rust::Vec<int64_t> dims, rust::String s_dtype, rust::String s_layout, rust::String s_device, bool requires_grad, bool pin_memory);
rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor>& tensor);
rust::String dtype(const std::shared_ptr<CrossTensor> &tensor);
