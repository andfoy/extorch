#include "extorch/include/wrapper.h"
#include "extorch/src/native.rs.h"
#include "extorch/include/utils.h"
#include <iostream>


std::shared_ptr<CrossTensor> unsqueeze(const std::shared_ptr<CrossTensor> &tensor, int64_t dim) {
    CrossTensor cross_tensor = *tensor.get();
    auto ret_tensor = cross_tensor.unsqueeze(dim);
    return std::make_shared<CrossTensor>(std::move(ret_tensor));
}
