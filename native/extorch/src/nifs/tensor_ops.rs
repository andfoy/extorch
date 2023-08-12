use crate::native::torch;
use crate::shared_types::{TensorStruct, TensorIndex};

use rustler::{Error, NifResult};

nif_impl!(unsqueeze, TensorStruct<'a>, tensor: TensorStruct<'a>, dim: i64);
nif_impl!(index, TensorStruct<'a>, tensor: TensorStruct<'a>, indices: TensorIndex);
