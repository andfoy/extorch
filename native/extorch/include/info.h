
#include "common.h"
#include "utils.h"

rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor> &tensor);
int64_t dim(const std::shared_ptr<CrossTensor> &tensor);
rust::String dtype(const std::shared_ptr<CrossTensor> &tensor);
rust::String memory_format(const std::shared_ptr<CrossTensor> &tensor);
rust::String layout(const std::shared_ptr<CrossTensor> &tensor);
Device device(const std::shared_ptr<CrossTensor> &tensor);
rust::String repr(const std::shared_ptr<CrossTensor> &tensor, const PrintOptions opts);
ScalarList to_list(const std::shared_ptr<CrossTensor> &tensor);
Scalar item(const std::shared_ptr<CrossTensor> &tensor);
bool requires_grad(const std::shared_ptr<CrossTensor> &tensor);
int64_t numel(const std::shared_ptr<CrossTensor> &tensor);
bool is_complex(const std::shared_ptr<CrossTensor> &tensor);
bool is_floating_point(const std::shared_ptr<CrossTensor> &tensor);
bool is_conj(const std::shared_ptr<CrossTensor> &tensor);
bool is_nonzero(const std::shared_ptr<CrossTensor> &tensor);

std::shared_ptr<CrossTensor> to(
    const std::shared_ptr<CrossTensor> &tensor,
    rust::String dtype, Device device,
    bool non_blocking, bool copy,
    rust::String memory_format);
