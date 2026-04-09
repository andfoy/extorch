#include "extorch/src/native.rs.h"
#include "extorch/include/pointwise.h"


std::shared_ptr<CrossTensor> real(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor in_tensor = *input.get();
    torch::Tensor tensor = torch::real(in_tensor);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> imag(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor in_tensor = *input.get();
    torch::Tensor tensor = torch::imag(in_tensor);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

// Arithmetic

std::shared_ptr<CrossTensor> add(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other, Scalar alpha) {
    auto a = get_scalar_type(alpha);
    torch::Tensor tensor = torch::add(*input, *other, a);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> sub(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other, Scalar alpha) {
    auto a = get_scalar_type(alpha);
    torch::Tensor tensor = torch::sub(*input, *other, a);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> mul(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other) {
    torch::Tensor tensor = torch::mul(*input, *other);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> tensor_div(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other) {
    torch::Tensor tensor = torch::div(*input, *other);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> neg(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = torch::neg(*input);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> tensor_abs(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = torch::abs(*input);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> pow_tensor(const std::shared_ptr<CrossTensor> &input, Scalar exponent) {
    auto e = get_scalar_type(exponent);
    torch::Tensor tensor = torch::pow(*input, e);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

// Math

std::shared_ptr<CrossTensor> tensor_exp(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = torch::exp(*input);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> tensor_log(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = torch::log(*input);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> tensor_sqrt(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = torch::sqrt(*input);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> tensor_sin(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = torch::sin(*input);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> tensor_cos(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = torch::cos(*input);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> clamp(const std::shared_ptr<CrossTensor> &input, Scalar min_val, Scalar max_val) {
    auto mn = get_scalar_type(min_val);
    auto mx = get_scalar_type(max_val);
    torch::Tensor tensor = torch::clamp(*input, mn, mx);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

// Linear algebra

std::shared_ptr<CrossTensor> matmul(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other) {
    torch::Tensor tensor = torch::matmul(*input, *other);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> mm(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other) {
    torch::Tensor tensor = torch::mm(*input, *other);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> bmm(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other) {
    torch::Tensor tensor = torch::bmm(*input, *other);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

// Conditional / masking

std::shared_ptr<CrossTensor> tensor_where(const std::shared_ptr<CrossTensor> &condition, const std::shared_ptr<CrossTensor> &x, const std::shared_ptr<CrossTensor> &y) {
    torch::Tensor tensor = torch::where(*condition, *x, *y);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> masked_fill(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &mask, Scalar value) {
    auto v = get_scalar_type(value);
    torch::Tensor in_tensor = *input;
    torch::Tensor tensor = in_tensor.masked_fill(*mask, v);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

// Tensor manipulation

std::shared_ptr<CrossTensor> contiguous(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = input->contiguous();
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> clone(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = input->clone();
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> detach(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = input->detach();
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> view(const std::shared_ptr<CrossTensor> &input, rust::Vec<int64_t> shape) {
    const int64_t *ptr = shape.data();
    torch::Tensor tensor = input->view(torch::IntArrayRef{ptr, shape.size()});
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> expand(const std::shared_ptr<CrossTensor> &input, rust::Vec<int64_t> shape) {
    const int64_t *ptr = shape.data();
    torch::Tensor tensor = input->expand(torch::IntArrayRef{ptr, shape.size()});
    return std::make_shared<CrossTensor>(std::move(tensor));
}

// Functional activations

std::shared_ptr<CrossTensor> functional_softmax(const std::shared_ptr<CrossTensor> &input, int64_t dim) {
    torch::Tensor tensor = torch::softmax(*input, dim);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> functional_log_softmax(const std::shared_ptr<CrossTensor> &input, int64_t dim) {
    torch::Tensor tensor = torch::log_softmax(*input, dim);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> functional_relu(const std::shared_ptr<CrossTensor> &input) {
    torch::Tensor tensor = torch::relu(*input);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

// Einsum

std::shared_ptr<CrossTensor> einsum(rust::String equation, const std::shared_ptr<CrossTensor> &a, const std::shared_ptr<CrossTensor> &b) {
    std::string eq(equation);
    torch::Tensor tensor = torch::einsum(eq, {*a, *b});
    return std::make_shared<CrossTensor>(std::move(tensor));
}
