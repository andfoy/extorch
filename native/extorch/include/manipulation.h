
#include "common.h"
#include "utils.h"

std::shared_ptr<CrossTensor> unsqueeze(
    const std::shared_ptr<CrossTensor> &tensor,
    int64_t dim);

std::shared_ptr<CrossTensor> index(
    const std::shared_ptr<CrossTensor> &tensor, const rust::Vec<TorchIndex>);

std::shared_ptr<CrossTensor> index_put(
    const std::shared_ptr<CrossTensor> &tensor,
    const rust::Vec<TorchIndex> index,
    const std::shared_ptr<CrossTensor> &value);
