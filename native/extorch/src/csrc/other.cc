#include "extorch/src/native.rs.h"
#include "extorch/include/other.h"


std::shared_ptr<CrossTensor> view_as_complex(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor in_tensor = *input.get();
    torch::Tensor tensor = torch::view_as_complex(in_tensor);
    return std::make_shared<CrossTensor>(std::move(tensor));
}
