
#include "common.h"
#include "utils.h"

rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor> &tensor);
rust::String dtype(const std::shared_ptr<CrossTensor> &tensor);
rust::String memory_format(const std::shared_ptr<CrossTensor> &tensor);
rust::String layout(const std::shared_ptr<CrossTensor> &tensor);
Device device(const std::shared_ptr<CrossTensor> &tensor);
rust::String repr(const std::shared_ptr<CrossTensor> &tensor, const PrintOptions opts);
ScalarList to_list(const std::shared_ptr<CrossTensor> &tensor);
bool requires_grad(const std::shared_ptr<CrossTensor> &tensor);
int64_t numel(const std::shared_ptr<CrossTensor> &tensor);
