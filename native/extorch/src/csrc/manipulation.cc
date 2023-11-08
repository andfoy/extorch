#include "extorch/src/native.rs.h"
#include "extorch/include/utils.h"


std::shared_ptr<CrossTensor> unsqueeze(const std::shared_ptr<CrossTensor> &tensor, int64_t dim) {
    CrossTensor cross_tensor = *tensor.get();
    auto ret_tensor = cross_tensor.unsqueeze(dim);
    return std::make_shared<CrossTensor>(std::move(ret_tensor));
}

std::shared_ptr<CrossTensor> reshape(const std::shared_ptr<CrossTensor> &tensor, rust::Vec<int64_t> shape) {
    CrossTensor cross_tensor = *tensor.get();
    const int64_t *ptr = shape.data();
    auto ret_tensor = torch::reshape(cross_tensor, torch::IntArrayRef{ptr, shape.size()});
    return std::make_shared<CrossTensor>(std::move(ret_tensor));
}


std::shared_ptr<CrossTensor> index(const std::shared_ptr<CrossTensor> &tensor, const rust::Vec<TorchIndex> index) {
    CrossTensor cross_tensor = *tensor.get();
    std::vector<at::indexing::TensorIndex> act_index;
    for(int i = 0; i < index.size(); i++) {
        const TorchIndex& idx = index[i];
        at::indexing::TensorIndexType idx_type = static_cast<at::indexing::TensorIndexType>(idx.type_);
        if(idx_type == at::indexing::TensorIndexType::None) {
            act_index.push_back(at::indexing::TensorIndex(c10::nullopt));
        } else if(idx_type == at::indexing::TensorIndexType::Ellipsis) {
            act_index.push_back(at::indexing::TensorIndex(at::indexing::Ellipsis));
        } else if(idx.type_ == 2) {
            act_index.push_back(idx.integer);
        } else if(idx_type == at::indexing::TensorIndexType::Boolean) {
            act_index.push_back(idx.boolean);
        } else if(idx_type == at::indexing::TensorIndexType::Slice) {
            const TorchSlice& ex_slice = idx.slice;
            c10::optional<c10::SymInt> start = c10::nullopt;
            c10::optional<c10::SymInt> stop = c10::nullopt;
            c10::optional<c10::SymInt> step = c10::nullopt;

            uint8_t mask = ex_slice.enc;
            if(mask & 1) {
                start = ex_slice.start;
            }
            mask >>= 1;
            if(mask & 1) {
                stop = ex_slice.stop;
            }
            mask >>= 1;
            if(mask & 1) {
                step = ex_slice.step;
            }
            at::indexing::Slice act_slice(start, stop, step);
            act_index.push_back(act_slice);
        } else if(idx_type == at::indexing::TensorIndexType::Tensor) {
            CrossTensor cross_tensor_idx = *idx.tensor.get();
            act_index.push_back(cross_tensor_idx);
        }
    }
    auto sliced_tensor = cross_tensor.index(act_index);
    // auto sliced_tensor = cross_tensor[0];
    return std::make_shared<CrossTensor>(std::move(sliced_tensor));
}

std::shared_ptr<CrossTensor> index_put(
        const std::shared_ptr<CrossTensor> &tensor,
        const rust::Vec<TorchIndex> index,
        const std::shared_ptr<CrossTensor> &value,
        bool inplace) {
    CrossTensor out_tensor;
    CrossTensor cross_tensor = *tensor.get();
    CrossTensor value_tensor = *value.get();

    std::vector<at::indexing::TensorIndex> act_index;
    for(int i = 0; i < index.size(); i++) {
        const TorchIndex& idx = index[i];
        at::indexing::TensorIndexType idx_type = static_cast<at::indexing::TensorIndexType>(idx.type_);
        if(idx_type == at::indexing::TensorIndexType::None) {
            act_index.push_back(at::indexing::TensorIndex(c10::nullopt));
        } else if(idx_type == at::indexing::TensorIndexType::Ellipsis) {
            act_index.push_back(at::indexing::TensorIndex(at::indexing::Ellipsis));
        } else if(idx.type_ == 2) {
            act_index.push_back(idx.integer);
        } else if(idx_type == at::indexing::TensorIndexType::Boolean) {
            act_index.push_back(idx.boolean);
        } else if(idx_type == at::indexing::TensorIndexType::Slice) {
            const TorchSlice& ex_slice = idx.slice;
            c10::optional<c10::SymInt> start = c10::nullopt;
            c10::optional<c10::SymInt> stop = c10::nullopt;
            c10::optional<c10::SymInt> step = c10::nullopt;

            uint8_t mask = ex_slice.enc;
            if(mask & 1) {
                start = ex_slice.start;
            }
            mask >>= 1;
            if(mask & 1) {
                stop = ex_slice.stop;
            }
            mask >>= 1;
            if(mask & 1) {
                step = ex_slice.step;
            }
            at::indexing::Slice act_slice(start, stop, step);
            act_index.push_back(act_slice);
        } else if(idx_type == at::indexing::TensorIndexType::Tensor) {
            CrossTensor cross_tensor_idx = *idx.tensor.get();
            act_index.push_back(cross_tensor_idx);
        }
    }

    // const int64_t *ptr = list.size.data();
    // torch::detail::TensorDataContainer scalar_list = get_scalar_list(std::move(list.list));
    // torch::TensorOptions opts = cross_tensor.options();
    // opts = opts.requires_grad(torch::nullopt);
    // torch::Tensor value_tensor = torch::tensor(scalar_list, opts);
    // value_tensor = value_tensor.reshape(torch::IntArrayRef{ptr, list.size.size()}).contiguous();
    if(!inplace) {
        out_tensor = cross_tensor.clone();
    } else {
        out_tensor = cross_tensor;
    }

    out_tensor = out_tensor.index_put_(act_index, value_tensor);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> conj(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor in_tensor = *input.get();
    torch::Tensor tensor = torch::conj(in_tensor);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

 std::shared_ptr<CrossTensor> adjoint(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor in_tensor = *input.get();
    torch::Tensor tensor = torch::adjoint(in_tensor);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> transpose(
        const std::shared_ptr<CrossTensor> &input, int64_t dim0, int64_t dim1) {
    CrossTensor in_tensor = *input.get();
    torch::Tensor tensor = torch::transpose(in_tensor, dim0, dim1);
    return std::make_shared<CrossTensor>(std::move(tensor));
}

std::shared_ptr<CrossTensor> cat(TensorList seq, int64_t dim, TensorOut opt_out) {
    std::vector<CrossTensor> tensor_seq = unpack_tensor_list(seq);
    CrossTensor out_tensor;

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::cat_out(out_tensor, tensor_seq, dim);
    } else {
        out_tensor = torch::cat(tensor_seq, dim);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

TensorList chunk(
        const std::shared_ptr<CrossTensor> &input, int64_t chunks, int64_t dim) {
    CrossTensor in_tensor = *input.get();
    std::vector<CrossTensor> seq = torch::chunk(in_tensor, chunks, dim);
    return pack_tensor_list(seq);
}

TensorList tensor_split(
        const std::shared_ptr<CrossTensor> &input, TensorOrInt indices_or_sections,
        int64_t dim) {

    CrossTensor in_tensor = *input.get();
    std::vector<CrossTensor> seq;

    if(indices_or_sections.is_tensor) {
        CrossTensor sections = *indices_or_sections.tensor.get();
        seq = torch::tensor_split(in_tensor, sections, dim);
    } else {
        seq = torch::tensor_split(in_tensor, indices_or_sections.value, dim);
    }
    return pack_tensor_list(seq);
}

TensorList dsplit(
        const std::shared_ptr<CrossTensor> &input, IntListOrInt indices_or_sections) {

    CrossTensor in_tensor = *input.get();
    std::vector<CrossTensor> seq;

    if(indices_or_sections.is_list) {
        rust::Vec<int64_t> sections = indices_or_sections.list;
        const int64_t *ptr = sections.data();
        seq = torch::dsplit(in_tensor, torch::IntArrayRef{ptr, sections.size()});
    } else {
        seq = torch::dsplit(in_tensor, indices_or_sections.value);
    }
    return pack_tensor_list(seq);
}

std::shared_ptr<CrossTensor> column_stack(TensorList tensor_list, TensorOut opt_out) {
    std::vector<CrossTensor> tensor_vec = unpack_tensor_list(tensor_list);
    CrossTensor out_tensor;

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::column_stack_out(out_tensor, tensor_vec);
    } else {
        out_tensor = torch::column_stack(tensor_vec);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> dstack(TensorList tensor_list, TensorOut opt_out) {
    std::vector<CrossTensor> tensor_vec = unpack_tensor_list(tensor_list);
    CrossTensor out_tensor;

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::dstack_out(out_tensor, tensor_vec);
    } else {
        out_tensor = torch::dstack(tensor_vec);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> gather(
        const std::shared_ptr<CrossTensor> &input, int64_t dim,
        const std::shared_ptr<CrossTensor> &index, bool sparse_grad,
        TensorOut opt_out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor index_tensor = *index.get();

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::gather_out(out_tensor, in_tensor, dim, index_tensor, sparse_grad);
    } else {
        out_tensor = torch::gather(in_tensor, dim, index_tensor, sparse_grad);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

TensorList hsplit(
        const std::shared_ptr<CrossTensor> &input, IntListOrInt indices_or_sections) {

    CrossTensor in_tensor = *input.get();
    std::vector<CrossTensor> seq;

    if(indices_or_sections.is_list) {
        rust::Vec<int64_t> sections = indices_or_sections.list;
        const int64_t *ptr = sections.data();
        seq = torch::hsplit(in_tensor, torch::IntArrayRef{ptr, sections.size()});
    } else {
        seq = torch::hsplit(in_tensor, indices_or_sections.value);
    }
    return pack_tensor_list(seq);
}

std::shared_ptr<CrossTensor> hstack(TensorList tensor_list, TensorOut opt_out) {
    std::vector<CrossTensor> tensor_vec = unpack_tensor_list(tensor_list);
    CrossTensor out_tensor;

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::hstack_out(out_tensor, tensor_vec);
    } else {
        out_tensor = torch::hstack(tensor_vec);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> index_add(
        const std::shared_ptr<CrossTensor> &tensor,
        int64_t dim,
        const std::shared_ptr<CrossTensor> &index,
        const std::shared_ptr<CrossTensor> &source,
        Scalar s_scalar,
        TensorOut out,
        bool inplace) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *tensor.get();
    CrossTensor index_tensor = *index.get();
    CrossTensor source_tensor = *source.get();

    torch::Scalar scalar = get_scalar_type(s_scalar);

    if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::index_add_out(
            out_tensor, in_tensor, dim, index_tensor, source_tensor, scalar);
    } else if(!inplace) {
        out_tensor = torch::index_add(
            in_tensor, dim, index_tensor, source_tensor, scalar);
    } else {
        out_tensor = in_tensor;
        out_tensor.index_add_(dim, index_tensor, source_tensor, scalar);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> index_copy(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        const std::shared_ptr<CrossTensor> &index,
        const std::shared_ptr<CrossTensor> &source,
        TensorOut out,
        bool inplace) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor index_tensor = *index.get();
    CrossTensor source_tensor = *source.get();

    if(inplace) {
        out_tensor = in_tensor;
        out_tensor.index_copy_(dim, index_tensor, source_tensor);
    } else if (out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::index_copy_out(
            out_tensor, in_tensor, dim, index_tensor, source_tensor);
    } else {
        out_tensor = torch::index_copy(in_tensor, dim, index_tensor, source_tensor);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> index_reduce(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        const std::shared_ptr<CrossTensor> &index,
        const std::shared_ptr<CrossTensor> &source,
        rust::String s_reduce,
        bool include_self,
        TensorOut out,
        bool inplace) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor index_tensor = *index.get();
    CrossTensor source_tensor = *source.get();
    std::string reduce(s_reduce.data(), s_reduce.size());

    if(inplace) {
        out_tensor = in_tensor;
        out_tensor.index_reduce_(
            dim, index_tensor, source_tensor, reduce, include_self);
    } else if (out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::index_reduce_out(
            out_tensor, in_tensor, dim, index_tensor, source_tensor,
            reduce, include_self);
    } else {
        out_tensor = torch::index_reduce(
            in_tensor, dim, index_tensor, source_tensor, reduce, include_self);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> index_select(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        const std::shared_ptr<CrossTensor> &index,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor index_tensor = *index.get();

    if (out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::index_select_out(
            out_tensor, in_tensor, dim, index_tensor);
    } else {
        out_tensor = torch::index_select(in_tensor, dim, index_tensor);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> masked_select(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &mask,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor mask_tensor = *mask.get();

    if (out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::masked_select_out(
            out_tensor, in_tensor, mask_tensor);
    } else {
        out_tensor = torch::masked_select(in_tensor, mask_tensor);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> movedim(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> source,
        rust::Vec<int64_t> destination) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::movedim(
        in_tensor, torch::IntArrayRef{source.data(), source.size()},
        torch::IntArrayRef{destination.data(), destination.size()});

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}
