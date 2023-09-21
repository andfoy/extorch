
#include "common.h"
#include "utils.h"

std::shared_ptr<CrossTensor> all(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorOut opt_out);

std::shared_ptr<CrossTensor> any(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorOut opt_out);

std::shared_ptr<CrossTensor> argmax(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim);

std::shared_ptr<CrossTensor> argmin(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim);

TensorTuple max(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out);

TensorTuple min(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out);
