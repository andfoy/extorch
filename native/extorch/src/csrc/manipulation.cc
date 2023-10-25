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
        const std::shared_ptr<CrossTensor> &value) {
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

    auto out_tensor = cross_tensor.index_put_(act_index, value_tensor);
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
