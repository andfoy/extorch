use crate::native::torch;
use crate::shared_types::TensorStruct;

use rustler::{Error, NifResult};

nif_impl!(conj, TensorStruct<'a>, input: TensorStruct<'a>);
