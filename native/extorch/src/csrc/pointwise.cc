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
