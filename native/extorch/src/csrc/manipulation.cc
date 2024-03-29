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

std::shared_ptr<CrossTensor> narrow(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        TensorOrInt start,
        int64_t length) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();

    if(start.is_tensor) {
        CrossTensor start_tensor = *start.tensor.get();
        if(start_tensor.numel() == 1) {
            start_tensor = start_tensor.squeeze();
        }
        out_tensor = torch::narrow(in_tensor, dim, start_tensor, length);
    } else {
        out_tensor = torch::narrow(in_tensor, dim, start.value, length);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> narrow_copy(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        int64_t start,
        int64_t length,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::narrow_copy_out(
            out_tensor, in_tensor, dim, start, length);
    } else {
        out_tensor = torch::narrow_copy(in_tensor, dim, start, length);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

TensorTuple nonzero(
        const std::shared_ptr<CrossTensor> &input,
        TensorOut out,
        bool as_tuple) {

    std::vector<std::shared_ptr<CrossTensor>> out_tensor_vec;
    CrossTensor in_tensor = *input.get();

    if(!as_tuple) {
        CrossTensor out_tensor;
        if(out.used) {
            out_tensor = *out.tensor.get();
            out_tensor = torch::nonzero_out(out_tensor, in_tensor);
        } else {
            out_tensor = torch::nonzero(in_tensor);
        }
        out_tensor_vec.push_back(
            std::make_shared<CrossTensor>(std::move(out_tensor)));
    } else  {
        std::vector<CrossTensor> out_tensors = torch::nonzero_numpy(in_tensor);
        for(int i = 0; i < out_tensors.size(); i++) {
            out_tensor_vec.push_back(
                std::make_shared<CrossTensor>(std::move(out_tensors[i])));
        }
    }

    return pack_tensor_tuple(out_tensor_vec);
}

std::shared_ptr<CrossTensor> permute(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::permute(
        in_tensor, torch::IntArrayRef{dims.data(), dims.size()});

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> vstack(TensorList tensor_list, TensorOut opt_out) {
    std::vector<CrossTensor> tensor_vec = unpack_tensor_list(tensor_list);
    CrossTensor out_tensor;

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::vstack_out(out_tensor, tensor_vec);
    } else {
        out_tensor = torch::vstack(tensor_vec);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> select(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        int64_t index) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::select(in_tensor, dim, index);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> scatter(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        const std::shared_ptr<CrossTensor> &index,
        const std::shared_ptr<CrossTensor> &src,
        TensorOut out,
        const bool inplace) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor index_tensor = *index.get();
    CrossTensor src_tensor = *src.get();

    if(inplace) {
        out_tensor = in_tensor;
        out_tensor.scatter_(dim, index_tensor, src_tensor);
    } else if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::scatter_out(
            out_tensor, in_tensor, dim, index_tensor, src_tensor);
    } else {
        out_tensor = torch::scatter(in_tensor, dim, index_tensor, src_tensor);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> diagonal_scatter(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &src,
        int64_t offset,
        int64_t dim1,
        int64_t dim2,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor src_tensor = *src.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::diagonal_scatter_out(
            out_tensor, in_tensor, src_tensor, offset, dim1, dim2);
    } else {
        out_tensor = torch::diagonal_scatter(
            in_tensor, src_tensor, offset, dim1, dim2);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> select_scatter(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &src,
        int64_t dim,
        int64_t index,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor src_tensor = *src.get();

    if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::select_scatter_out(out_tensor, in_tensor, src_tensor, dim, index);
    } else {
        out_tensor = torch::select_scatter(in_tensor, src_tensor, dim, index);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> slice_scatter(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &src,
        int64_t dim,
        OptionalInt start,
        OptionalInt end,
        int64_t step,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor src_tensor = *src.get();

    torch::optional<int64_t> start_opt = torch::nullopt;
    torch::optional<int64_t> end_opt = torch::nullopt;

    if(start.used) {
        start_opt = start.value;
    }

    if(end.used) {
        end_opt = end.value;
    }

    if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::slice_scatter_out(
            out_tensor, in_tensor, src_tensor, dim, start_opt, end_opt, step);
    } else {
        out_tensor = torch::slice_scatter(
            in_tensor, src_tensor, dim, start_opt, end_opt, step);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> scatter_add(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        const std::shared_ptr<CrossTensor> &index,
        const std::shared_ptr<CrossTensor> &src,
        TensorOut out,
        const bool inplace) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor index_tensor = *index.get();
    CrossTensor src_tensor = *src.get();

    if(inplace) {
        out_tensor = in_tensor;
        out_tensor.scatter_add_(dim, index_tensor, src_tensor);
    } else if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::scatter_add_out(
            out_tensor, in_tensor, dim, index_tensor, src_tensor);
    } else {
        out_tensor = torch::scatter_add(in_tensor, dim, index_tensor, src_tensor);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> scatter_reduce(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        const std::shared_ptr<CrossTensor> &index,
        const std::shared_ptr<CrossTensor> &src,
        rust::String s_reduce,
        bool include_self,
        TensorOut out,
        const bool inplace) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor index_tensor = *index.get();
    CrossTensor src_tensor = *src.get();
    std::string reduce(s_reduce.data(), s_reduce.size());

    if(inplace) {
        out_tensor = in_tensor;
        out_tensor.scatter_reduce_(dim, index_tensor, src_tensor, reduce, include_self);
    } else if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::scatter_reduce_out(
            out_tensor, in_tensor, dim, index_tensor, src_tensor, reduce, include_self);
    } else {
        out_tensor = torch::scatter_reduce(
            in_tensor, dim, index_tensor, src_tensor, reduce, include_self);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

TensorList split(
        const std::shared_ptr<CrossTensor> &input,
        IntListOrInt indices_or_sections, int64_t dim) {

    CrossTensor in_tensor = *input.get();
    std::vector<CrossTensor> seq;

    if(indices_or_sections.is_list) {
        rust::Vec<int64_t> sections = indices_or_sections.list;
        const int64_t *ptr = sections.data();
        seq = torch::split(in_tensor, torch::IntArrayRef{ptr, sections.size()}, dim);
    } else {
        seq = torch::split(in_tensor, indices_or_sections.value, dim);
    }
    return pack_tensor_list(seq);
}

std::shared_ptr<CrossTensor> squeeze(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims) {
    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    if(dims.size() == 0) {
        out_tensor = torch::squeeze(in_tensor);
    } else {
        out_tensor = torch::squeeze(in_tensor, torch::IntArrayRef{dims.data(), dims.size()});
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> stack(TensorList seq, int64_t dim, TensorOut opt_out) {
    CrossTensor out_tensor;
    std::vector<CrossTensor> tensor_seq = unpack_tensor_list(seq);

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::stack_out(out_tensor, tensor_seq, dim);
    } else {
        out_tensor = torch::stack(tensor_seq, dim);
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> t(const std::shared_ptr<CrossTensor> &input) {
    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    out_tensor = torch::t(in_tensor);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> take(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &indices) {
    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor indices_tensor = *indices.get();

    out_tensor = torch::take(in_tensor, indices_tensor);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> take_along_dim(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &indices,
        OptionalInt dim,
        TensorOut out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor indices_tensor = *indices.get();

    torch::optional<int64_t> opt_dim = torch::nullopt;
    if(dim.used) {
        opt_dim = dim.value;
    }

    if(out.used) {
        out_tensor = *out.tensor.get();
        out_tensor = torch::take_along_dim_out(out_tensor, in_tensor, indices_tensor, opt_dim);
    } else {
        out_tensor = torch::take_along_dim(in_tensor, indices_tensor, opt_dim);
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}
