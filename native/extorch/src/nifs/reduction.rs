use crate::native::torch;
use crate::native::torch::{OptionalInt, TensorOut};
use crate::shared_types::TensorStruct;

use rustler::{Error, NifResult};


nif_impl!(
    all,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    out: TensorOut
);

nif_impl!(
    any,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    out: TensorOut
);
