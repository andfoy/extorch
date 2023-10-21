
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

std::shared_ptr<CrossTensor> amax(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim, TensorOut opt_out
);

std::shared_ptr<CrossTensor> amin(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim, TensorOut opt_out
);

TensorTuple aminmax(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out
);

std::shared_ptr<CrossTensor> dist(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        Scalar p
);

std::shared_ptr<CrossTensor> logsumexp(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim, TensorOut opt_out
);

std::shared_ptr<CrossTensor> sum(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim
);

std::shared_ptr<CrossTensor> nansum(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim
);

std::shared_ptr<CrossTensor> mean(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim,
        TensorOut opt_out
);

std::shared_ptr<CrossTensor> nanmean(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim,
        TensorOut opt_out
);

TensorTuple median(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out
);

TensorTuple nanmedian(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out
);

TensorTuple mode(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim, bool keepdim, TensorTuple opt_out
);

std::shared_ptr<CrossTensor> prod(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim
);

std::shared_ptr<CrossTensor> quantile(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &q,
        OptionalInt opt_dim, bool keepdim,
        rust::String interpolation,
        TensorOut out);

std::shared_ptr<CrossTensor> nanquantile(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &q,
        OptionalInt opt_dim, bool keepdim,
        rust::String interpolation,
        TensorOut out);

std::shared_ptr<CrossTensor> std_dev(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, int64_t correction, bool keepdim,
        TensorOut opt_out
);

TensorTuple std_mean(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, int64_t correction,
        bool keepdim, TensorTuple opt_out
);

TensorTuple unique(
        const std::shared_ptr<CrossTensor> &input,
        bool sorted, bool return_inverse, bool return_counts,
        OptionalInt dim);
