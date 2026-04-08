#include "extorch/src/native.rs.h"
#include "extorch/include/info.h"
#include "extorch/include/printing.h"


rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor> &tensor)
{
    CrossTensor cross_tensor = *tensor.get();
    auto sizes = cross_tensor.sizes();
    rust::Slice<const int64_t> slice{sizes.data(), sizes.size()};
    return slice;
}

int64_t dim(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.dim();
}

rust::String dtype(const std::shared_ptr<CrossTensor> &tensor)
{
    CrossTensor cross_tensor = *tensor.get();
    auto type = cross_tensor.dtype().toScalarType();
    auto type_name = inv_type_mapping[type];
    rust::String type_rust(type_name.data(), type_name.size());
    return type_rust;
}

Device device(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto device = cross_tensor.device();
    auto device_type = device.type();
    auto device_index = device.index();
    auto it = std::find_if(
        device_mapping.begin(), device_mapping.end(),
        [&device_type](const std::pair<std::string, torch::DeviceType> &p) {
            return p.second == device_type;
        });
    auto device_name = it->first;
    rust::String device_rust(device_name.data(), device_name.size());
    Device this_device { device_rust, device_index };
    return this_device;
}

rust::String repr(const std::shared_ptr<CrossTensor> &tensor, const PrintOptions opts) {
    CrossTensor cross_tensor = *tensor.get();
    auto str_repr = _tensor_str(cross_tensor, opts, 0);
    // auto str_repr = cross_tensor.toString();
    // std::stringstream ss;
    // ss << cross_tensor;
    // auto str_repr = ss.str();
    rust::String tensor_repr(str_repr.data(), str_repr.size());
    return tensor_repr;
}

template<typename scalar_t>
Scalar pack_complex_scalar(scalar_t scalar, torch::ScalarType type) {
    Scalar scalar_s;
    std::string identifier = "complex32";
    if(type == torch::ScalarType::ComplexFloat) {
        identifier = "complex64";
    } else if (type == torch::ScalarType::ComplexDouble) {
        identifier = "complex128";
    }

    auto re = static_cast<double>(scalar.real());
    auto imag = static_cast<double>(scalar.imag());

    auto re_repr = static_cast<uint8_t*>(static_cast<void*>(&re));
    auto im_repr = static_cast<uint8_t*>(static_cast<void*>(&imag));

    rust::Vec<uint8_t> vec_repr;
    std::copy(re_repr, re_repr + sizeof(double), std::back_inserter(vec_repr));
    std::copy(im_repr, im_repr + sizeof(double), std::back_inserter(vec_repr));

    rust::string entry_used(identifier.data(), identifier.size());
    scalar_s.entry_used = entry_used;
    scalar_s._repr = vec_repr;
    return scalar_s;
}

template<typename scalar_t>
Scalar pack_scalar(scalar_t scalar, torch::ScalarType type) {
    Scalar scalar_s;
    std::string identifier = "float32";
    if(type == torch::ScalarType::Bool) {
        // scalar_s._bool = (bool) scalar;
        identifier = "bool";
    } else if(type == torch::ScalarType::Byte) {
        // scalar_s._ui8 = scalar;
        identifier = "uint8";
    } else if(type == torch::ScalarType::Char) {
        // scalar_s._i8 = scalar;
        identifier = "int8";
    } else if(type == torch::ScalarType::Short) {
        // scalar_s._i16 = scalar;
        identifier = "int16";
    } else if(type == torch::ScalarType::Int) {
        // scalar_s._i32 = scalar;
        identifier = "int32";
    } else if(type == torch::ScalarType::Long) {
        // scalar_s._i64 = scalar;
        identifier = "int64";
    } else if(type == torch::ScalarType::Half) {
        // scalar_s._f16 = (float) scalar;
        identifier = "float32";
    } else if(type == torch::ScalarType::Float) {
        // scalar_s._f32 = scalar;
        identifier = "float32";
    } else if(type == torch::ScalarType::Double) {
        // scalar_s._f64 = scalar;
        identifier = "float64";
    }
    // } else if(type == torch::ScalarType::ComplexHalf) {
    //     return pack_complex_scalar<c10::Half>(scalar, type);
    // } else if(type == torch::ScalarType::ComplexFloat) {
    //     return pack_complex_scalar<float>(scalar, type);
    // } else if (type == torch::ScalarType::ComplexDouble) {
    //     return pack_complex_scalar<double>(scalar, type);
    // }

    auto scalar_repr = static_cast<uint8_t*>(static_cast<void*>(&scalar));
    rust::Vec<uint8_t> vec_repr;
    std::copy(scalar_repr, scalar_repr + sizeof(scalar_t), std::back_inserter(vec_repr));

    rust::string entry_used(identifier.data(), identifier.size());
    scalar_s.entry_used = entry_used;
    scalar_s._repr = vec_repr;
    return scalar_s;
}

ScalarList to_list(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto conv_tensor = cross_tensor.contiguous().to(torch::kCPU);
    auto size = cross_tensor.sizes().vec();

    rust::Vec<int64_t> rust_size;
    for(auto dim: size) {
        rust_size.push_back(dim);
    }

    auto numel = cross_tensor.numel();

    rust::Vec<Scalar> scalar_list;

    AT_DISPATCH_SWITCH(cross_tensor.scalar_type(), "to_list",
        AT_DISPATCH_CASE_ALL_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::Bool,
            [&] {
                auto data_ptr = conv_tensor.data_ptr<scalar_t>();
                for(int64_t i = 0; i < numel; i++) {
                    auto scalar = pack_scalar<scalar_t>(data_ptr[i], cross_tensor.scalar_type());
                    scalar_list.push_back(scalar);
                }
        })
        AT_DISPATCH_CASE_COMPLEX_TYPES_AND(torch::ScalarType::ComplexHalf,
            [&] {
                auto data_ptr = conv_tensor.data_ptr<scalar_t>();
                for(int64_t i = 0; i < numel; i++) {
                    auto scalar = pack_complex_scalar<scalar_t>(data_ptr[i], cross_tensor.scalar_type());
                    scalar_list.push_back(scalar);
                }
        })
    );

    ScalarList result {
        std::move(scalar_list),
        std::move(rust_size)
    };
    // result.list = ;
    // result.size = rust_size;
    return result;
}

Scalar item(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto scalar_value = cross_tensor.item();
    Scalar scalar;

    AT_DISPATCH_SWITCH(cross_tensor.scalar_type(), "to_list",
        AT_DISPATCH_CASE_ALL_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::Bool,
        [&] {
            scalar = pack_scalar<scalar_t>(scalar_value.to<scalar_t>(), cross_tensor.scalar_type());
        })
        AT_DISPATCH_CASE_COMPLEX_TYPES_AND(torch::ScalarType::ComplexHalf,
        [&] {
            scalar = pack_complex_scalar<scalar_t>(scalar_value.to<scalar_t>(), cross_tensor.scalar_type());
        })
    );

    return std::move(scalar);
}

bool requires_grad(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.requires_grad();
}

int64_t numel(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.numel();
}

rust::String memory_format(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto type = cross_tensor.suggest_memory_format();
    auto type_name = inv_memory_fmt_mapping[type];
    rust::String type_rust(type_name.data(), type_name.size());
    return type_rust;
}

rust::String layout(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto type = cross_tensor.layout();
    auto type_name = inv_layout_mapping[type];
    rust::String type_rust(type_name.data(), type_name.size());
    return type_rust;
}

bool is_complex(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_complex();
}

bool is_floating_point(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_floating_point();
}

bool is_conj(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_conj();
}

bool is_nonzero(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_nonzero();
}

std::shared_ptr<CrossTensor> to(
        const std::shared_ptr<CrossTensor> &tensor,
        rust::String dtype, Device device,
        bool non_blocking, bool copy,
        rust::String memory_format) {

    CrossTensor out_tensor;
    CrossTensor cross_tensor = *tensor.get();
    torch::TensorOptions opts = get_tensor_options(
        dtype, "", device, false, false, memory_format);

    out_tensor = cross_tensor.to(
        opts.device(), opts.dtype().toScalarType(), non_blocking, copy,
        opts.memory_format_opt());

    return std::make_shared<CrossTensor>(std::move(out_tensor));
}

// ============================================================================
// CUDA memory monitoring
// ============================================================================

bool cuda_is_available() {
    return torch::cuda::is_available();
}

int64_t cuda_device_count() {
    return static_cast<int64_t>(torch::cuda::device_count());
}

int64_t cuda_memory_allocated(int64_t device_index) {
#ifdef USE_CUDA
    if (!torch::cuda::is_available()) return -1;
    return static_cast<int64_t>(c10::cuda::CUDACachingAllocator::currentMemoryAllocated(device_index));
#else
    (void)device_index;
    return -1;
#endif
}

int64_t cuda_memory_reserved(int64_t device_index) {
#ifdef USE_CUDA
    if (!torch::cuda::is_available()) return -1;
    return static_cast<int64_t>(c10::cuda::CUDACachingAllocator::currentMemoryCached(device_index));
#else
    (void)device_index;
    return -1;
#endif
}

int64_t cuda_max_memory_allocated(int64_t device_index) {
#ifdef USE_CUDA
    if (!torch::cuda::is_available()) return -1;
    return static_cast<int64_t>(c10::cuda::CUDACachingAllocator::maxMemoryAllocated(device_index));
#else
    (void)device_index;
    return -1;
#endif
}

// ============================================================================
// Zero-copy tensor exchange
// ============================================================================

int64_t data_ptr(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return reinterpret_cast<int64_t>(cross_tensor.data_ptr());
}

rust::Vec<int64_t> strides(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    auto s = cross_tensor.strides();
    rust::Vec<int64_t> result;
    for (auto v : s) {
        result.push_back(v);
    }
    return result;
}

int64_t element_size(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.element_size();
}

bool is_contiguous(const std::shared_ptr<CrossTensor> &tensor) {
    CrossTensor cross_tensor = *tensor.get();
    return cross_tensor.is_contiguous();
}

std::shared_ptr<CrossTensor> from_blob(
    int64_t ptr,
    rust::Vec<int64_t> shape,
    rust::Vec<int64_t> strides_vec,
    rust::String s_dtype,
    Device s_device)
{
    void *data = reinterpret_cast<void *>(ptr);
    std::string dtype_str(s_dtype);
    auto dtype = type_mapping[dtype_str];

    const int64_t *shape_ptr = shape.data();
    auto shape_ref = torch::IntArrayRef{shape_ptr, shape.size()};

    const int64_t *strides_ptr = strides_vec.data();
    auto strides_ref = torch::IntArrayRef{strides_ptr, strides_vec.size()};

    // from_blob does NOT take ownership of the memory.
    // The caller is responsible for keeping the source alive.
    auto opts = torch::TensorOptions().dtype(dtype);
    torch::Tensor tensor = torch::from_blob(data, shape_ref, strides_ref, opts);

    return std::make_shared<CrossTensor>(tensor);
}
