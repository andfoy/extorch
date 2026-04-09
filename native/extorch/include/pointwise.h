
#include "common.h"
#include "utils.h"

std::shared_ptr<CrossTensor> real(
    const std::shared_ptr<CrossTensor> &input);

std::shared_ptr<CrossTensor> imag(
    const std::shared_ptr<CrossTensor> &input);

// Arithmetic
std::shared_ptr<CrossTensor> add(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other, Scalar alpha);
std::shared_ptr<CrossTensor> sub(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other, Scalar alpha);
std::shared_ptr<CrossTensor> mul(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other);
std::shared_ptr<CrossTensor> tensor_div(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other);
std::shared_ptr<CrossTensor> neg(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> tensor_abs(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> pow_tensor(const std::shared_ptr<CrossTensor> &input, Scalar exponent);

std::shared_ptr<CrossTensor> clamp(const std::shared_ptr<CrossTensor> &input, Scalar min_val, Scalar max_val);

// Math
std::shared_ptr<CrossTensor> tensor_exp(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> tensor_log(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> tensor_sqrt(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> tensor_sin(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> tensor_cos(const std::shared_ptr<CrossTensor> &input);

// Linear algebra
std::shared_ptr<CrossTensor> matmul(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other);
std::shared_ptr<CrossTensor> mm(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other);
std::shared_ptr<CrossTensor> bmm(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &other);

// Conditional / masking
std::shared_ptr<CrossTensor> tensor_where(const std::shared_ptr<CrossTensor> &condition, const std::shared_ptr<CrossTensor> &x, const std::shared_ptr<CrossTensor> &y);
std::shared_ptr<CrossTensor> masked_fill(const std::shared_ptr<CrossTensor> &input, const std::shared_ptr<CrossTensor> &mask, Scalar value);

// Tensor manipulation
std::shared_ptr<CrossTensor> contiguous(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> clone(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> detach(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> view(const std::shared_ptr<CrossTensor> &input, rust::Vec<int64_t> shape);
std::shared_ptr<CrossTensor> expand(const std::shared_ptr<CrossTensor> &input, rust::Vec<int64_t> shape);

// Functional activations
std::shared_ptr<CrossTensor> functional_softmax(const std::shared_ptr<CrossTensor> &input, int64_t dim);
std::shared_ptr<CrossTensor> functional_log_softmax(const std::shared_ptr<CrossTensor> &input, int64_t dim);
std::shared_ptr<CrossTensor> functional_relu(const std::shared_ptr<CrossTensor> &input);

// Einsum
std::shared_ptr<CrossTensor> einsum(rust::String equation, const std::shared_ptr<CrossTensor> &a, const std::shared_ptr<CrossTensor> &b);
