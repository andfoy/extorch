
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

std::shared_ptr<CrossTensor> masked_select(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &mask,
    TensorOut out);

std::shared_ptr<CrossTensor> movedim(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> source,
    rust::Vec<int64_t> destination);

std::shared_ptr<CrossTensor> narrow(
    const std::shared_ptr<CrossTensor> &input,
    int64_t dim,
    TensorOrInt start,
    int64_t length);

std::shared_ptr<CrossTensor> narrow_copy(
    const std::shared_ptr<CrossTensor> &input,
    int64_t dim,
    int64_t start,
    int64_t length,
    TensorOut out);

TensorTuple nonzero(
    const std::shared_ptr<CrossTensor> &input,
    TensorOut out,
    bool as_tuple);

std::shared_ptr<CrossTensor> permute(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> dims);

std::shared_ptr<CrossTensor> vstack(TensorList tensor_list, TensorOut opt_out);

std::shared_ptr<CrossTensor> select(
    const std::shared_ptr<CrossTensor> &input,
    int64_t dim,
    int64_t index);

std::shared_ptr<CrossTensor> scatter(
    const std::shared_ptr<CrossTensor> &input,
    int64_t dim,
    const std::shared_ptr<CrossTensor> &index,
    const std::shared_ptr<CrossTensor> &src,
    TensorOut out,
    const bool inplace);

std::shared_ptr<CrossTensor> diagonal_scatter(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &src,
    int64_t offset,
    int64_t dim1,
    int64_t dim2,
    TensorOut out);

std::shared_ptr<CrossTensor> select_scatter(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &src,
    int64_t dim,
    int64_t index,
    TensorOut out);

std::shared_ptr<CrossTensor> slice_scatter(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &src,
    int64_t dim,
    OptionalInt start,
    OptionalInt end,
    int64_t step,
    TensorOut out);

std::shared_ptr<CrossTensor> scatter_add(
    const std::shared_ptr<CrossTensor> &input,
    int64_t dim,
    const std::shared_ptr<CrossTensor> &index,
    const std::shared_ptr<CrossTensor> &src,
    TensorOut out,
    const bool inplace);
