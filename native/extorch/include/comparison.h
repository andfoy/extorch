
#include "common.h"
#include "utils.h"

bool allclose(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        double rtol, double atol, bool equal_nan);

std::shared_ptr<CrossTensor> argsort(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        bool descending,
        bool stable);

SortResult sort(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        bool descending,
        bool stable,
        SortResult out_r);

std::shared_ptr<CrossTensor> eq(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

bool equal(const std::shared_ptr<CrossTensor> &input,
           const std::shared_ptr<CrossTensor> &other);

std::shared_ptr<CrossTensor> ge(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

std::shared_ptr<CrossTensor> gt(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

std::shared_ptr<CrossTensor> le(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

std::shared_ptr<CrossTensor> lt(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

std::shared_ptr<CrossTensor> ne(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

std::shared_ptr<CrossTensor> isclose(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        double rtol, double atol, bool equal_nan);

std::shared_ptr<CrossTensor> isfinite(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> isinf(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> isposinf(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> isneginf(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> isnan(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> isreal(const std::shared_ptr<CrossTensor> &input);

std::shared_ptr<CrossTensor> isin(
        const std::shared_ptr<CrossTensor> &elements,
        const std::shared_ptr<CrossTensor> &test_elements,
        bool assume_unique, bool invert);

SortResult kthvalue(
        const std::shared_ptr<CrossTensor> &input,
        int64_t k,
        int64_t dim,
        bool keepdim,
        SortResult out_r);

std::shared_ptr<CrossTensor> maximum(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

std::shared_ptr<CrossTensor> minimum(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

std::shared_ptr<CrossTensor> fmax(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);

std::shared_ptr<CrossTensor> fmin(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out);


SortResult topk(
        const std::shared_ptr<CrossTensor> &input,
        int64_t k,
        int64_t dim,
        bool largest,
        bool sorted,
        SortResult out_r
);
