use crate::native::torch;
use crate::shared_types::{Size, TensorOptions, TensorStruct};
use crate::torch::{Scalar, ScalarList};

use rustler::{Error, NifResult};

// trace_macros!(true);
nif_impl!(empty, TensorStruct<'a>, size: Size, options: TensorOptions);
nif_impl!(zeros, TensorStruct<'a>, size: Size, options: TensorOptions);
nif_impl!(ones, TensorStruct<'a>, size: Size, options: TensorOptions);
nif_impl!(rand, TensorStruct<'a>, size: Size, options: TensorOptions);
nif_impl!(randn, TensorStruct<'a>, size: Size, options: TensorOptions);

nif_impl!(
    randint,
    TensorStruct<'a>,
    low: i64,
    high: i64,
    size: Size,
    options: TensorOptions
);
nif_impl!(
    full,
    TensorStruct<'a>,
    size: Size,
    scalar: Scalar,
    options: TensorOptions
);
nif_impl!(
    eye,
    TensorStruct<'a>,
    n: i64,
    m: i64,
    options: TensorOptions
);
nif_impl!(
    arange,
    TensorStruct<'a>,
    start: Scalar,
    end: Scalar,
    step: Scalar,
    options: TensorOptions
);
nif_impl!(
    linspace,
    TensorStruct<'a>,
    start: Scalar,
    end: Scalar,
    steps: i64,
    options: TensorOptions
);

nif_impl!(
    logspace,
    TensorStruct<'a>,
    start: Scalar,
    end: Scalar,
    steps: i64,
    base: Scalar,
    options: TensorOptions
);

nif_impl!(
    tensor,
    TensorStruct<'a>,
    list: ScalarList,
    options: TensorOptions
);

nif_impl!(
    complex,
    TensorStruct<'a>,
    real: TensorStruct<'a>,
    imag: TensorStruct<'a>
);

nif_impl!(
    polar,
    TensorStruct<'a>,
    abs: TensorStruct<'a>,
    angle: TensorStruct<'a>
);

nif_impl!(
    empty_like,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    options: TensorOptions
);

nif_impl!(
    rand_like,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    options: TensorOptions
);

nif_impl!(
    randn_like,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    options: TensorOptions
);

nif_impl!(
    randint_like,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    low: i64,
    high: i64,
    options: TensorOptions
);

nif_impl!(
    full_like,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    value: Scalar,
    options: TensorOptions
);

nif_impl!(
    zeros_like,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    options: TensorOptions
);

nif_impl!(
    ones_like,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    options: TensorOptions
);
