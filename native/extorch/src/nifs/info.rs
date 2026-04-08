use crate::native::torch;
use crate::native::torch::{Device, PrintOptions};
use crate::shared_types::{AtomString, Size, TensorStruct};

use rustler::{Error, NifResult};

// trace_macros!(true);
nif_impl!(repr, String, tensor: TensorStruct<'a>, opts: PrintOptions);
nif_impl!(size, Size, tensor: TensorStruct<'a>);
nif_impl!(dim, i64, tensor: TensorStruct<'a>);
nif_impl!(device, torch::Device, tensor: TensorStruct<'a>);
nif_impl!(dtype, AtomString, tensor: TensorStruct<'a>);
nif_impl!(requires_grad, bool, tensor: TensorStruct<'a>);
nif_impl!(memory_format, AtomString, tensor: TensorStruct<'a>);
nif_impl!(layout, AtomString, tensor: TensorStruct<'a>);
nif_impl!(to_list, torch::ScalarList, tensor: TensorStruct<'a>);
nif_impl!(item, torch::Scalar, tensor: TensorStruct<'a>);
nif_impl!(numel, i64, tensor: TensorStruct<'a>);
nif_impl!(is_complex, bool, tensor: TensorStruct<'a>);
nif_impl!(is_floating_point, bool, tensor: TensorStruct<'a>);
nif_impl!(is_conj, bool, tensor: TensorStruct<'a>);
nif_impl!(is_nonzero, bool, tensor: TensorStruct<'a>);
// nif_impl!(repr, String, tensor => Tensor);
nif_impl!(
    to,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    i_dtype: AtomString,
    i_device: Device,
    non_blocking: bool,
    copy: bool,
    i_memory_format: AtomString
);

// CUDA memory monitoring
nif_impl!(cuda_is_available, bool);
nif_impl!(cuda_device_count, i64);
nif_impl!(cuda_memory_allocated, i64, device_index: i64);
nif_impl!(cuda_memory_reserved, i64, device_index: i64);
nif_impl!(cuda_max_memory_allocated, i64, device_index: i64);

// Zero-copy tensor exchange
nif_impl!(data_ptr, i64, tensor: TensorStruct<'a>);
nif_impl!(strides, Vec<i64>, tensor: TensorStruct<'a>);
nif_impl!(element_size, i64, tensor: TensorStruct<'a>);
nif_impl!(is_contiguous, bool, tensor: TensorStruct<'a>);
nif_impl!(
    from_blob,
    TensorStruct<'a>,
    ptr: i64,
    shape: Size,
    blob_strides: Size,
    s_dtype: AtomString,
    s_device: Device
);
