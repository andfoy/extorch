#include "extorch/src/native.rs.h"
#include "extorch/include/utils.h"


std::shared_ptr<CrossTensor> unsqueeze(const std::shared_ptr<CrossTensor> &tensor, int64_t dim) {
    CrossTensor cross_tensor = *tensor.get();
    auto ret_tensor = cross_tensor.unsqueeze(dim);
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
        } else if(idx_type == at::indexing::TensorIndexType::Integer) {
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