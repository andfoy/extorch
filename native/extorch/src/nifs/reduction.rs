use crate::native::torch;
use crate::native::torch::{OptionalInt, TensorOut, TensorTuple, Scalar};
use crate::shared_types::{TensorStruct, Size};

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

nif_impl!(
    amax,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    keepdim: bool,
    out: TensorOut
);

nif_impl!(
    amin,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    keepdim: bool,
    out: TensorOut
);

nif_impl!(
    aminmax,
    TensorTuple,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    out: TensorTuple
);

nif_impl!(
    dist,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    p: Scalar
);
