
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

// Zero-copy tensor exchange
int64_t data_ptr(const std::shared_ptr<CrossTensor> &tensor);
rust::Vec<int64_t> strides(const std::shared_ptr<CrossTensor> &tensor);
int64_t element_size(const std::shared_ptr<CrossTensor> &tensor);
bool is_contiguous(const std::shared_ptr<CrossTensor> &tensor);

// CUDA memory monitoring
bool cuda_is_available();
int64_t cuda_device_count();
int64_t cuda_memory_allocated(int64_t device_index);
int64_t cuda_memory_reserved(int64_t device_index);
int64_t cuda_max_memory_allocated(int64_t device_index);

std::shared_ptr<CrossTensor> from_blob(
    int64_t ptr,
    rust::Vec<int64_t> shape,
    rust::Vec<int64_t> strides,
    rust::String s_dtype,
    Device s_device);
