#include "extorch/src/native.rs.h"
#include "extorch/include/mutating.h"

std::shared_ptr<CrossTensor> conj(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor in_tensor = *input.get();
    torch::Tensor tensor = torch::conj(in_tensor);
    return std::make_shared<CrossTensor>(std::move(tensor));
}
