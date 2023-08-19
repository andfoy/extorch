use crate::native::torch;
use crate::shared_types::TensorStruct;

use rustler::{Error, NifResult};

nif_impl!(real, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(imag, TensorStruct<'a>, input: TensorStruct<'a>);
