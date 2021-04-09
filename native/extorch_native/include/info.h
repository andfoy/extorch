
#include "common.h"
#include "utils.h"

rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor> &tensor);
rust::String dtype(const std::shared_ptr<CrossTensor> &tensor);
Device device(const std::shared_ptr<CrossTensor> &tensor);
rust::String repr(const std::shared_ptr<CrossTensor> &tensor);
ScalarList to_list(const std::shared_ptr<CrossTensor> &tensor);
