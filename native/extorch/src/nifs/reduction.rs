use crate::native::torch;
use crate::native::torch::{OptionalInt, TensorOut, TensorTuple, Scalar};
use crate::shared_types::{TensorStruct, Size, AtomString};

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

nif_impl!(
    logsumexp,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    keepdim: bool,
    out: TensorOut
);

nif_impl!(
    sum,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    keepdim: bool
);

nif_impl!(
    nansum,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    keepdim: bool
);

nif_impl!(
    mean,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    keepdim: bool,
    out: TensorOut
);

nif_impl!(
    nanmean,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    keepdim: bool,
    out: TensorOut
);

nif_impl!(
    median,
    TensorTuple,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    out: TensorTuple
);

nif_impl!(
    nanmedian,
    TensorTuple,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    out: TensorTuple
);

nif_impl!(
    mode,
    TensorTuple,
    input: TensorStruct<'a>,
    dim: i64,
    keepdim: bool,
    out: TensorTuple
);

nif_impl!(
    prod,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool
);

nif_impl!(
    quantile,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    q: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    interpolation: AtomString,
    out: TensorOut
);

nif_impl!(
    nanquantile,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    q: TensorStruct<'a>,
    dim: OptionalInt,
    keepdim: bool,
    interpolation: AtomString,
    out: TensorOut
);

nif_impl!(
    std_dev,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    correction: i64,
    keepdim: bool,
    out: TensorOut
);

nif_impl!(
    std_mean,
    TensorTuple,
    input: TensorStruct<'a>,
    dim: Size,
    correction: i64,
    keepdim: bool,
    out: TensorTuple
);

nif_impl!(
    unique,
    TensorTuple,
    input: TensorStruct<'a>,
    sorted: bool,
    return_inverse: bool,
    return_counts: bool,
    dim: OptionalInt
);

nif_impl!(
    unique_consecutive,
    TensorTuple,
    input: TensorStruct<'a>,
    return_inverse: bool,
    return_counts: bool,
    dim: OptionalInt
);

nif_impl!(
    var,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size,
    correction: i64,
    keepdim: bool,
    out: TensorOut
);

nif_impl!(
    var_mean,
    TensorTuple,
    input: TensorStruct<'a>,
    dim: Size,
    correction: i64,
    keepdim: bool,
    out: TensorTuple
);

nif_impl!(
    count_nonzero,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: Size
);
