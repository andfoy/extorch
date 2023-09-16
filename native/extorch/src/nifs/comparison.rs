use crate::native::torch::SortResult;
use crate::shared_types::TensorStruct;
use crate::{native::torch, torch::TensorOut};

use rustler::{Error, NifResult};

nif_impl!(
    allclose,
    bool,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    rtol: f64,
    atol: f64,
    equal_nan: bool
);

nif_impl!(
    argsort,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: i64,
    descending: bool,
    stable: bool
);

nif_impl!(
    sort,
    SortResult,
    input: TensorStruct<'a>,
    dim: i64,
    descending: bool,
    stable: bool,
    out: SortResult
);

nif_impl!(
    eq,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    out: TensorOut
);

nif_impl!(
    equal,
    bool,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>
);

nif_impl!(
    ge,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    out: TensorOut
);

nif_impl!(
    gt,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    out: TensorOut
);

nif_impl!(
    le,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    out: TensorOut
);

nif_impl!(
    lt,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    out: TensorOut
);

nif_impl!(
    ne,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    out: TensorOut
);

nif_impl!(
    isclose,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    rtol: f64,
    atol: f64,
    equal_nan: bool
);

nif_impl!(isfinite, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(isinf, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(isposinf, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(isneginf, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(isnan, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(isreal, TensorStruct<'a>, input: TensorStruct<'a>);

nif_impl!(
    isin,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    other: TensorStruct<'a>,
    assume_unique: bool,
    invert: bool
);

nif_impl!(
    kthvalue,
    SortResult,
    input: TensorStruct<'a>,
    k: i64,
    dim: i64,
    keepdim: bool,
    out: SortResult
);
