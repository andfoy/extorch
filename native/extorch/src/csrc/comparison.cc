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

