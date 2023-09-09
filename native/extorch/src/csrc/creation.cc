#include "extorch/src/native.rs.h"
#include "extorch/include/creation.h"


std::shared_ptr<CrossTensor> empty(
    rust::Vec<int64_t> dims, rust::String s_dtype,
    rust::String s_layout, Device s_device,
    bool requires_grad, bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::empty(torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> zeros(
    rust::Vec<int64_t> dims, rust::String s_dtype,
    rust::String s_layout, Device s_device,
    bool requires_grad, bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::zeros(torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> ones(
    rust::Vec<int64_t> dims, rust::String s_dtype,
    rust::String s_layout, Device s_device,
    bool requires_grad, bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::ones(torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> full(
    rust::Vec<int64_t> dims,
    Scalar scalar,
    rust::String s_dtype,
    rust::String s_layout,
    Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    auto torch_scalar = get_scalar_type(scalar);
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::full(torch::IntArrayRef{ptr, dims.size()}, torch_scalar, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> rand(
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::rand(torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> randn(
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::randn(torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> randint(
    int64_t low,
    int64_t high,
    rust::Vec<int64_t> dims,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::randint(low, high, torch::IntArrayRef{ptr, dims.size()}, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> eye(
    int64_t n,
    int64_t m,
    rust::String s_dtype,
    rust::String s_layout,
    Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    // auto torch_scalar = get_scalar_type(scalar);
    // const int64_t *ptr = dims.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::eye(n, m, opts);
    // torch::Tensor tensor = torch::full(torch::IntArrayRef{ptr, dims.size()}, torch_scalar, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> arange(
    struct Scalar start,
    struct Scalar end,
    struct Scalar step,
    rust::String s_dtype,
    rust::String s_layout,
    Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    // auto torch_scalar = get_scalar_type(scalar);
    // const int64_t *ptr = dims.data();
    auto start_scalar = get_scalar_type(start);
    auto end_scalar = get_scalar_type(end);
    auto step_scalar = get_scalar_type(step);
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::arange(start_scalar, end_scalar, step_scalar, opts);
    // torch::Tensor tensor = torch::full(torch::IntArrayRef{ptr, dims.size()}, torch_scalar, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> linspace(
    struct Scalar start,
    struct Scalar end,
    int64_t steps,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt)
{
    auto start_scalar = get_scalar_type(start);
    auto end_scalar = get_scalar_type(end);
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);

    torch::Tensor tensor = torch::linspace(start_scalar, end_scalar, steps, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

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
    rust::String s_mem_fmt)
{
    auto start_scalar = get_scalar_type(start);
    auto end_scalar = get_scalar_type(end);
    auto base_scalar = get_scalar_type(base);

    auto base_double = base_scalar.toDouble();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    torch::Tensor tensor = torch::logspace(start_scalar, end_scalar, steps, base_double, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> tensor(
    ScalarList list,
    rust::String s_dtype,
    rust::String s_layout,
    struct Device s_device,
    bool requires_grad,
    bool pin_memory,
    rust::String s_mem_fmt
)
{
    const int64_t *ptr = list.size.data();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    auto scalar_type = opts.dtype().toScalarType();
    torch::detail::TensorDataContainer scalar_list;

    if(torch::isComplexType(scalar_type)) {
        scalar_list = get_complex_tensor_parts(std::move(list.list), scalar_type, opts, ptr);
    } else {
        scalar_list = get_scalar_list(std::move(list.list));
    }

    opts = opts.requires_grad(torch::nullopt);
    torch::Tensor tensor = torch::tensor(scalar_list, opts);
    torch::Tensor reshaped_tensor = tensor;
    if(tensor.numel() > 0) {
        reshaped_tensor = tensor.reshape(torch::IntArrayRef{ptr, list.size.size()}).contiguous();
        reshaped_tensor.set_requires_grad(requires_grad);
    }
    return std::make_shared<CrossTensor>(std::move(reshaped_tensor));
}

std::shared_ptr<CrossTensor> complex(
        const std::shared_ptr<CrossTensor> &real,
        const std::shared_ptr<CrossTensor> &imag) {
    CrossTensor real_tensor = *real.get();
    CrossTensor imag_tensor = *imag.get();
    torch::Tensor tensor = torch::complex(real_tensor, imag_tensor);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> polar(
        const std::shared_ptr<CrossTensor> &abs,
        const std::shared_ptr<CrossTensor> &angle) {
    CrossTensor abs_tensor = *abs.get();
    CrossTensor angle_tensor = *angle.get();
    torch::Tensor tensor = torch::polar(abs_tensor, angle_tensor);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> empty_like(
        const std::shared_ptr<CrossTensor> &input,
        rust::String s_dtype,
        rust::String s_layout,
        struct Device s_device,
        bool requires_grad,
        bool pin_memory,
        rust::String s_mem_fmt) {
    CrossTensor in_tensor = *input.get();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    torch::Tensor tensor = torch::empty_like(in_tensor, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> rand_like(
        const std::shared_ptr<CrossTensor> &input,
        rust::String s_dtype,
        rust::String s_layout,
        struct Device s_device,
        bool requires_grad,
        bool pin_memory,
        rust::String s_mem_fmt) {
    CrossTensor in_tensor = *input.get();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    torch::Tensor tensor = torch::rand_like(in_tensor, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> randn_like(
        const std::shared_ptr<CrossTensor> &input,
        rust::String s_dtype,
        rust::String s_layout,
        struct Device s_device,
        bool requires_grad,
        bool pin_memory,
        rust::String s_mem_fmt) {
    CrossTensor in_tensor = *input.get();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    torch::Tensor tensor = torch::randn_like(in_tensor, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> randint_like(
        const std::shared_ptr<CrossTensor> &input,
        int64_t low,
        int64_t high,
        rust::String s_dtype,
        rust::String s_layout,
        struct Device s_device,
        bool requires_grad,
        bool pin_memory,
        rust::String s_mem_fmt) {
    CrossTensor in_tensor = *input.get();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    torch::Tensor tensor = torch::randint_like(in_tensor, low, high, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> full_like(
        const std::shared_ptr<CrossTensor> &input,
        Scalar scalar,
        rust::String s_dtype,
        rust::String s_layout,
        struct Device s_device,
        bool requires_grad,
        bool pin_memory,
        rust::String s_mem_fmt) {
    CrossTensor in_tensor = *input.get();
    auto torch_scalar = get_scalar_type(scalar);
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    torch::Tensor tensor = torch::full_like(in_tensor, torch_scalar, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> zeros_like(
        const std::shared_ptr<CrossTensor> &input,
        rust::String s_dtype,
        rust::String s_layout,
        struct Device s_device,
        bool requires_grad,
        bool pin_memory,
        rust::String s_mem_fmt) {
    CrossTensor in_tensor = *input.get();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    torch::Tensor tensor = torch::zeros_like(in_tensor, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> ones_like(
        const std::shared_ptr<CrossTensor> &input,
        rust::String s_dtype,
        rust::String s_layout,
        struct Device s_device,
        bool requires_grad,
        bool pin_memory,
        rust::String s_mem_fmt) {
    CrossTensor in_tensor = *input.get();
    torch::TensorOptions opts = get_tensor_options(s_dtype, s_layout, s_device, requires_grad, pin_memory, s_mem_fmt);
    torch::Tensor tensor = torch::ones_like(in_tensor, opts);
    return std::make_shared<CrossTensor>(std::move(tensor));
}
