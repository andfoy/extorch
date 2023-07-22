use crate::native::torch;
use crate::shared_types::TensorStruct;

use rustler::{Error, NifResult};

nif_impl!(unsqueeze, TensorStruct<'a>, tensor: TensorStruct<'a>, dim: i64);
