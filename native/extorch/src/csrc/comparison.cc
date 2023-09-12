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
