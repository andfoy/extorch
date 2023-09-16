#include "extorch/src/native.rs.h"
#include "extorch/include/comparison.h"

bool allclose(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        double rtol, double atol, bool equal_nan) {
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();

    return torch::allclose(in_tensor, other_tensor, rtol, atol, equal_nan);
}

std::shared_ptr<CrossTensor> argsort(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        bool descending,
        bool stable) {

    CrossTensor in_tensor = *input.get();
    torch::Tensor out_tensor = torch::argsort(in_tensor, stable, dim, descending);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

SortResult sort(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        bool descending,
        bool stable,
        SortResult out_r) {

    CrossTensor values_tensor;
    CrossTensor indices_tensor;

    CrossTensor in_tensor = *input.get();
    if(out_r.used) {
        values_tensor = *out_r.values.get();
        indices_tensor = *out_r.indices.get();

        std::tie(values_tensor, indices_tensor) = torch::sort_out(
            values_tensor, indices_tensor, in_tensor, stable, dim, descending);
    } else {
        std::tie(values_tensor, indices_tensor) = torch::sort(
            in_tensor, stable, dim, descending);
    }

    auto values_out = std::make_shared<CrossTensor>(std::move(values_tensor));
    auto indices_out = std::make_shared<CrossTensor>(std::move(indices_tensor));
    return std::move(SortResult{values_out, indices_out, true});
}

std::shared_ptr<CrossTensor> eq(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        other_tensor = torch::eq_out(out_tensor, in_tensor, other_tensor);
    } else {
        out_tensor = torch::eq(in_tensor, other_tensor);
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

bool equal(const std::shared_ptr<CrossTensor> &input,
           const std::shared_ptr<CrossTensor> &other) {

    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();
    return torch::equal(in_tensor, other_tensor);
}

std::shared_ptr<CrossTensor> ge(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        other_tensor = torch::ge_out(out_tensor, in_tensor, other_tensor);
    } else {
        out_tensor = torch::ge(in_tensor, other_tensor);
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> gt(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        other_tensor = torch::gt_out(out_tensor, in_tensor, other_tensor);
    } else {
        out_tensor = torch::gt(in_tensor, other_tensor);
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> le(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        other_tensor = torch::le_out(out_tensor, in_tensor, other_tensor);
    } else {
        out_tensor = torch::le(in_tensor, other_tensor);
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> lt(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        other_tensor = torch::lt_out(out_tensor, in_tensor, other_tensor);
    } else {
        out_tensor = torch::lt(in_tensor, other_tensor);
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> ne(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        other_tensor = torch::ne_out(out_tensor, in_tensor, other_tensor);
    } else {
        out_tensor = torch::ne(in_tensor, other_tensor);
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> isclose(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        double rtol, double atol, bool equal_nan) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();
    out_tensor = torch::isclose(in_tensor, other_tensor, rtol, atol, equal_nan);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> isfinite(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::isfinite(in_tensor);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> isinf(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::isinf(in_tensor);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> isposinf(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::isposinf(in_tensor);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> isneginf(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::isneginf(in_tensor);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> isnan(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::isnan(in_tensor);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> isin(
        const std::shared_ptr<CrossTensor> &elements,
        const std::shared_ptr<CrossTensor> &test_elements,
        bool assume_unique, bool invert) {

    CrossTensor out_tensor;
    CrossTensor elements_tensor = *elements.get();
    CrossTensor test_tensor = *test_elements.get();
    out_tensor = torch::isin(elements_tensor, test_tensor, assume_unique, invert);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}
