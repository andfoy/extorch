use crate::conversion::{wrap_device, wrap_size, wrap_str_atom};
pub use crate::native::torch;
use crate::shared_types::TensorStruct;

use rustler::{Encoder, Env, Error, Term};

nif_impl!(size, Size, tensor => Tensor);
nif_impl!(device, Device, tensor => Tensor);
nif_impl!(dtype, StrAtom, tensor => Tensor);
nif_impl!(repr, String, tensor => Tensor);
