use crate::native::torch;
use crate::torch::Scalar;
use crate::shared_types::{Size, TensorOptions, TensorStruct};

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
nif_impl!(full, TensorStruct<'a>, size: Size, scalar: Scalar, options: TensorOptions);
nif_impl!(eye, TensorStruct<'a>, n: i64, m: i64, options: TensorOptions);

// trace_macros!(false);

// nif_impl!(empty, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(zeros, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(ones, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(rand, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(randn, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(randint, Tensor, low => i64, high => i64, sizes => Size, options => TensorOptions);
// nif_impl!(full, Tensor, sizes => Size, scalar => Scalar, options => TensorOptions);
// nif_impl!(eye, Tensor, n => i64, m => i64, options => TensorOptions);
// nif_impl!(arange, Tensor, start => Scalar, end => Scalar, step => Scalar, options => TensorOptions);
// nif_impl!(linspace, Tensor, start => Scalar, end => Scalar, steps => i64, options => TensorOptions);
// nif_impl!(logspace, Tensor, start => Scalar, end => Scalar, steps => i64, base => Scalar, options => TensorOptions);
// nif_impl!(tensor, Tensor, list => ListWrapper, options => TensorOptions);
