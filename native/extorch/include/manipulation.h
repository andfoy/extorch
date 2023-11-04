
#include "common.h"
#include "utils.h"

std::shared_ptr<CrossTensor> unsqueeze(
    const std::shared_ptr<CrossTensor> &tensor,
    int64_t dim);

std::shared_ptr<CrossTensor> reshape(
    const std::shared_ptr<CrossTensor> &tensor,
    rust::Vec<int64_t> shape);

std::shared_ptr<CrossTensor> index(
    const std::shared_ptr<CrossTensor> &tensor, const rust::Vec<TorchIndex>);

std::shared_ptr<CrossTensor> index_put(
    const std::shared_ptr<CrossTensor> &tensor,
    const rust::Vec<TorchIndex> index,
    const std::shared_ptr<CrossTensor> &value,
    bool inplace);

std::shared_ptr<CrossTensor> conj(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> adjoint(const std::shared_ptr<CrossTensor> &input);
std::shared_ptr<CrossTensor> transpose(
        const std::shared_ptr<CrossTensor> &input, int64_t dim0, int64_t dim1);

std::shared_ptr<CrossTensor> cat(TensorList seq, int64_t dim, TensorOut opt_out);
TensorList chunk(
        const std::shared_ptr<CrossTensor> &input, int64_t chunks, int64_t dim);
TensorList tensor_split(
        const std::shared_ptr<CrossTensor> &input, TensorOrInt indices_or_sections,
        int64_t dim);
TensorList dsplit(
        const std::shared_ptr<CrossTensor> &input, IntListOrInt indices_or_sections);

std::shared_ptr<CrossTensor> column_stack(TensorList tensor_list, TensorOut opt_out);
std::shared_ptr<CrossTensor> dstack(TensorList tensor_list, TensorOut opt_out);

std::shared_ptr<CrossTensor> gather(
        const std::shared_ptr<CrossTensor> &input, int64_t dim,
        const std::shared_ptr<CrossTensor> &index, bool sparse_grad,
        TensorOut opt_out);

TensorList hsplit(
        const std::shared_ptr<CrossTensor> &input, IntListOrInt indices_or_sections);
std::shared_ptr<CrossTensor> hstack(TensorList tensor_list, TensorOut opt_out);

std::shared_ptr<CrossTensor> index_add(
    const std::shared_ptr<CrossTensor> &tensor,
    int64_t dim,
    const std::shared_ptr<CrossTensor> &index,
    const std::shared_ptr<CrossTensor> &source,
    Scalar s_scalar,
    TensorOut out,
    bool inplace);

std::shared_ptr<CrossTensor> index_copy(
    const std::shared_ptr<CrossTensor> &input,
    int64_t dim,
    const std::shared_ptr<CrossTensor> &index,
    const std::shared_ptr<CrossTensor> &source,
    TensorOut out,
    bool inplace);

std::shared_ptr<CrossTensor> index_reduce(
    const std::shared_ptr<CrossTensor> &input,
    int64_t dim,
    const std::shared_ptr<CrossTensor> &index,
    const std::shared_ptr<CrossTensor> &source,
    rust::String s_reduce,
    bool include_self,
    TensorOut out,
    bool inplace);

std::shared_ptr<CrossTensor> index_select(
    const std::shared_ptr<CrossTensor> &input,
    int64_t dim,
    const std::shared_ptr<CrossTensor> &index,
    TensorOut out);
