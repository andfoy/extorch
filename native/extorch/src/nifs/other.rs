use crate::native::torch;
use crate::shared_types::TensorStruct;

use rustler::{Error, NifResult};

nif_impl!(view_as_complex, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(resolve_conj, TensorStruct<'a>, input: TensorStruct<'a>);
