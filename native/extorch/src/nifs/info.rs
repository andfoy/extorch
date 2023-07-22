use crate::native::torch;
use crate::shared_types::{AtomString, Size, TensorStruct};

use rustler::{Error, NifResult};

// trace_macros!(true);
nif_impl!(repr, String, tensor: TensorStruct<'a>);
nif_impl!(size, Size, tensor: TensorStruct<'a>);
nif_impl!(device, torch::Device, tensor: TensorStruct<'a>);
nif_impl!(dtype, AtomString, tensor: TensorStruct<'a>);
nif_impl!(to_list, torch::ScalarList, tensor: TensorStruct<'a>);
// nif_impl!(repr, String, tensor => Tensor);
