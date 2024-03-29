
#include "common.h"
#include "utils.h"

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

std::shared_ptr<CrossTensor> complex(
    const std::shared_ptr<CrossTensor> &real,
    const std::shared_ptr<CrossTensor> &imag);

std::shared_ptr<CrossTensor> polar(
    const std::shared_ptr<CrossTensor> &abs,
    const std::shared_ptr<CrossTensor> &angle);

std::shared_ptr<CrossTensor> empty_like(
    const std::shared_ptr<CrossTensor> &input,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> rand_like(
    const std::shared_ptr<CrossTensor> &input,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> randn_like(
    const std::shared_ptr<CrossTensor> &input,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> randint_like(
    const std::shared_ptr<CrossTensor> &input,
    int64_t low,
    int64_t high,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> full_like(
    const std::shared_ptr<CrossTensor> &input,
    Scalar scalar,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> zeros_like(
    const std::shared_ptr<CrossTensor> &input,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);

std::shared_ptr<CrossTensor> ones_like(
    const std::shared_ptr<CrossTensor> &input,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt);
