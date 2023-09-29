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

std::shared_ptr<CrossTensor> argmax(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    torch::optional<int64_t> dim = torch::nullopt;
    if(opt_dim.used) {
        dim = opt_dim.value;
    }
    out_tensor = torch::argmax(in_tensor, dim, keepdim);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}


std::shared_ptr<CrossTensor> argmin(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    torch::optional<int64_t> dim = torch::nullopt;
    if(opt_dim.used) {
        dim = opt_dim.value;
    }
    out_tensor = torch::argmin(in_tensor, dim, keepdim);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

TensorTuple max(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out) {

    std::vector<std::shared_ptr<CrossTensor>> out_vec;
    CrossTensor in_tensor = *input.get();

    if(opt_dim.used) {
        CrossTensor max;
        CrossTensor max_values;
        int64_t dim = opt_dim.value;

        if(opt_out.used) {
            std::vector<CrossTensor> tensor_out_list = unpack_tensor_tuple(opt_out, 2);
            max = tensor_out_list[0];
            max_values = tensor_out_list[1];
            std::tie<CrossTensor, CrossTensor>(max, max_values) = torch::max_out(
                max, max_values, in_tensor, dim, keepdim);
        } else {
            std::tie<CrossTensor, CrossTensor>(max, max_values) = torch::max(
                in_tensor, dim, keepdim);
        }
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(max)));
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(max_values)));

    } else {
        CrossTensor max;

        // Path disabled for consistency between max and min
        //
        // if(opt_out.used) {
        //     std::vector<CrossTensor> tensor_out_list = unpack_tensor_tuple(opt_out, 1);
        //     max = tensor_out_list[0];
        //     max = torch::max_out(max, in_tensor);
        // } else {
        //     max = torch::max(in_tensor);
        // }

        max = torch::max(in_tensor);
        out_vec.push_back(std::make_shared<CrossTensor>(max));

    }
    return pack_tensor_tuple(out_vec);
}

TensorTuple min(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out) {

    std::vector<std::shared_ptr<CrossTensor>> out_vec;
    CrossTensor in_tensor = *input.get();

    if(opt_dim.used) {
        CrossTensor min;
        CrossTensor min_values;
        int64_t dim = opt_dim.value;

        if(opt_out.used) {
            std::vector<CrossTensor> tensor_out_list = unpack_tensor_tuple(opt_out, 2);
            min = tensor_out_list[0];
            min_values = tensor_out_list[1];
            std::tie<CrossTensor, CrossTensor>(min, min_values) = torch::min_out(
                min, min_values, in_tensor, dim, keepdim);
        } else {
            std::tie<CrossTensor, CrossTensor>(min, min_values) = torch::min(
                in_tensor, dim, keepdim);
        }
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(min)));
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(min_values)));

    } else {
        CrossTensor min;
        min = torch::min(in_tensor);
        out_vec.push_back(std::make_shared<CrossTensor>(min));

    }
    return pack_tensor_tuple(out_vec);
}

std::shared_ptr<CrossTensor> amax(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim, TensorOut opt_out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    const int64_t *ptr = dims.data();

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::amax_out(out_tensor, in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    } else {
        out_tensor = torch::amax(in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> amin(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim, TensorOut opt_out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    const int64_t *ptr = dims.data();

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::amin_out(out_tensor, in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    } else {
        out_tensor = torch::amin(in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

TensorTuple aminmax(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out) {

    std::vector<std::shared_ptr<CrossTensor>> out_vec;
    CrossTensor in_tensor = *input.get();
    torch::optional<int64_t> dim = torch::nullopt;

    CrossTensor out_min;
    CrossTensor out_max;

    if(opt_dim.used) {
        dim = opt_dim.value;
    }

    if(opt_out.used) {
        std::vector<CrossTensor> tensor_out_list = unpack_tensor_tuple(opt_out, 2);
        out_min = tensor_out_list[0];
        out_max = tensor_out_list[1];

        std::tie<CrossTensor, CrossTensor>(out_min, out_max) = torch::aminmax_out(
            out_min, out_max, in_tensor, dim, keepdim);
    } else {
        std::tie<CrossTensor, CrossTensor>(out_min, out_max) = torch::aminmax(
            in_tensor, dim, keepdim);
    }

    out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_min)));
    out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_max)));
    return pack_tensor_tuple(out_vec);
}

std::shared_ptr<CrossTensor> dist(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        Scalar p) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    CrossTensor other_tensor = *other.get();
    auto torch_p = get_scalar_type(p);

    out_tensor = torch::dist(in_tensor, other_tensor, torch_p);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> logsumexp(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim, TensorOut opt_out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    const int64_t *ptr = dims.data();

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::logsumexp_out(
            out_tensor, in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    } else {
        out_tensor = torch::logsumexp(
            in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> sum(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    const int64_t *ptr = dims.data();

    out_tensor = torch::sum(in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> nansum(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    const int64_t *ptr = dims.data();

    out_tensor = torch::nansum(
        in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> mean(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim, TensorOut opt_out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    const int64_t *ptr = dims.data();

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::mean_out(
            out_tensor, in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    } else {
        out_tensor = torch::mean(in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

std::shared_ptr<CrossTensor> nanmean(
        const std::shared_ptr<CrossTensor> &input,
        rust::Vec<int64_t> dims, bool keepdim, TensorOut opt_out) {

    CrossTensor out_tensor;
    CrossTensor in_tensor = *input.get();
    const int64_t *ptr = dims.data();

    if(opt_out.used) {
        out_tensor = *opt_out.tensor.get();
        out_tensor = torch::nanmean_out(
            out_tensor, in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    } else {
        out_tensor = torch::nanmean(in_tensor, torch::IntArrayRef{ptr, dims.size()}, keepdim);
    }

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

TensorTuple median(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out) {

    std::vector<std::shared_ptr<CrossTensor>> out_vec;
    CrossTensor in_tensor = *input.get();
    int64_t dim = -1;

    CrossTensor out_values;
    CrossTensor out_indices;

    if(!opt_dim.used) {
        out_values = torch::median(in_tensor);
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_values)));
    } else {
        if(opt_out.used) {
            dim = opt_dim.value;
            std::vector<CrossTensor> tensor_out_list = unpack_tensor_tuple(opt_out, 2);
            out_values = tensor_out_list[0];
            out_indices = tensor_out_list[1];

            std::tie<CrossTensor, CrossTensor>(out_values, out_indices) = torch::median_out(
                out_values, out_indices, in_tensor, dim, keepdim);
        } else {
            std::tie<CrossTensor, CrossTensor>(out_values, out_indices) = torch::median(
                in_tensor, dim, keepdim);
        }
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_values)));
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_indices)));
    }

    return pack_tensor_tuple(out_vec);
}


TensorTuple nanmedian(
        const std::shared_ptr<CrossTensor> &input,
        OptionalInt opt_dim, bool keepdim, TensorTuple opt_out) {

    std::vector<std::shared_ptr<CrossTensor>> out_vec;
    CrossTensor in_tensor = *input.get();
    int64_t dim = -1;

    CrossTensor out_values;
    CrossTensor out_indices;

    if(!opt_dim.used) {
        out_values = torch::nanmedian(in_tensor);
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_values)));
    } else {
        if(opt_out.used) {
            dim = opt_dim.value;
            std::vector<CrossTensor> tensor_out_list = unpack_tensor_tuple(opt_out, 2);
            out_values = tensor_out_list[0];
            out_indices = tensor_out_list[1];

            std::tie<CrossTensor, CrossTensor>(out_values, out_indices) = torch::nanmedian_out(
                out_values, out_indices, in_tensor, dim, keepdim);
        } else {
            std::tie<CrossTensor, CrossTensor>(out_values, out_indices) = torch::nanmedian(
                in_tensor, dim, keepdim);
        }
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_values)));
        out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_indices)));
    }

    return pack_tensor_tuple(out_vec);
}

TensorTuple mode(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim, bool keepdim, TensorTuple opt_out) {

    std::vector<std::shared_ptr<CrossTensor>> out_vec;
    CrossTensor in_tensor = *input.get();
    CrossTensor out_values;
    CrossTensor out_indices;

    if(opt_out.used) {
        std::vector<CrossTensor> tensor_out_list = unpack_tensor_tuple(opt_out, 2);
        out_values = tensor_out_list[0];
        out_indices = tensor_out_list[1];

        std::tie<CrossTensor, CrossTensor>(out_values, out_indices) = torch::mode_out(
            out_values, out_indices, in_tensor, dim, keepdim);
    } else {
        std::tie<CrossTensor, CrossTensor>(out_values, out_indices) = torch::mode(
            in_tensor, dim, keepdim);
    }

    out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_values)));
    out_vec.push_back(std::make_shared<CrossTensor>(std::move(out_indices)));

    return pack_tensor_tuple(out_vec);
}
