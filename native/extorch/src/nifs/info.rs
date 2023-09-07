use crate::native::torch;
use crate::native::torch::PrintOptions;
use crate::shared_types::{AtomString, Size, TensorStruct};

use rustler::{Error, NifResult};

// trace_macros!(true);
nif_impl!(repr, String, tensor: TensorStruct<'a>, opts: PrintOptions);
nif_impl!(size, Size, tensor: TensorStruct<'a>);
nif_impl!(device, torch::Device, tensor: TensorStruct<'a>);
nif_impl!(dtype, AtomString, tensor: TensorStruct<'a>);
nif_impl!(requires_grad, bool, tensor: TensorStruct<'a>);
nif_impl!(memory_format, AtomString, tensor: TensorStruct<'a>);
nif_impl!(layout, AtomString, tensor: TensorStruct<'a>);
nif_impl!(to_list, torch::ScalarList, tensor: TensorStruct<'a>);
nif_impl!(numel, i64, tensor: TensorStruct<'a>);
// nif_impl!(repr, String, tensor => Tensor);
