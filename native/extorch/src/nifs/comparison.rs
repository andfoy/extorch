use crate::native::torch;
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