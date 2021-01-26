#pragma once
#include "rust/cxx.h"
#include <memory>
// #include <string>
#include <torch/torch.h>

// struct CrossTensorRef;

struct Device;
struct Scalar;
struct ScalarList;
using CrossTensor = torch::Tensor;

std::shared_ptr<CrossTensor> empty(
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> zeros(
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> ones(
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> full(
    rust::Vec<int64_t> dims,
    struct Scalar scalar,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);


std::shared_ptr<CrossTensor> rand(
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> randn(
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> randint(
    int64_t low,
    int64_t high,
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> eye(
    int64_t n,
    int64_t m,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> arange(
    struct Scalar start,
    struct Scalar end,
    struct Scalar step,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> linspace(
    struct Scalar start,
    struct Scalar end,
    int64_t steps,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> logspace(
    struct Scalar start,
    struct Scalar end,
    int64_t steps,
    struct Scalar base,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> tensor(
    struct ScalarList list,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt
);


rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor> &tensor);
rust::String dtype(const std::shared_ptr<CrossTensor> &tensor);
Device device(const std::shared_ptr<CrossTensor> &tensor);
rust::String repr(const std::shared_ptr<CrossTensor> &tensor);
