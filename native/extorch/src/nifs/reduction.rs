use crate::native::torch;
use crate::native::torch::{OptionalInt, TensorOut, TensorTuple};
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

nif_impl!(
    argmax,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool
);

nif_impl!(
    argmin,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool
);

nif_impl!(
    max,
    TensorTuple,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    out: TensorTuple
);

nif_impl!(
    min,
    TensorTuple,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    out: TensorTuple
);
