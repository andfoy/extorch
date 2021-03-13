use crate::conversion::wrap_tensor;
pub use crate::native::torch;
use crate::shared_types::TensorStruct;

use rustler::{Env, Error, Term};

nif_impl!(unsqueeze, Tensor, tensor => Tensor, dim => i64);
