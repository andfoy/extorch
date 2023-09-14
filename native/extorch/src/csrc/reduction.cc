#include "extorch/src/native.rs.h"
#include "extorch/include/reduction.h"

std::shared_ptr<CrossTensor> all(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorOut opt_out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    if(opt_dim.used) {
        int64_t dim = opt_dim.value;
        if(opt_out.used) {
            out_tensor = *opt_out.tensor.get();
            out_tensor = torch::all_out(out_tensor, in_tensor, dim, keepdim);
        } else {
            out_tensor = torch::all(in_tensor, dim, keepdim);
        }
    } else {
        if(opt_out.used) {
            out_tensor = *opt_out.tensor.get();
            out_tensor = torch::all_out(out_tensor, in_tensor);
        } else {
            out_tensor = torch::all(in_tensor);
        }
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> any(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorOut opt_out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    if(opt_dim.used) {
        int64_t dim = opt_dim.value;
        if(opt_out.used) {
            out_tensor = *opt_out.tensor.get();
            out_tensor = torch::any_out(out_tensor, in_tensor, dim, keepdim);
        } else {
            out_tensor = torch::any(in_tensor, dim, keepdim);
        }
    } else {
        if(opt_out.used) {
            out_tensor = *opt_out.tensor.get();
            out_tensor = torch::any_out(out_tensor, in_tensor);
        } else {
            out_tensor = torch::any(in_tensor);
        }
    }
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}
