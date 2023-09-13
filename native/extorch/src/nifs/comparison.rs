use crate::native::torch;
use crate::native::torch::SortResult;
use crate::shared_types::TensorStruct;

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
