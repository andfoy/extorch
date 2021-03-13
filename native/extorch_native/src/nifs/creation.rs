use crate::conversion::{
    unpack_list_wrapper, unpack_scalar, unpack_size_init, unpack_tensor_options, wrap_tensor,
};
use crate::native::torch;

use rustler::{Env, Error, Term};

nif_impl!(empty, Tensor, sizes => Size, options => TensorOptions);
nif_impl!(zeros, Tensor, sizes => Size, options => TensorOptions);
nif_impl!(ones, Tensor, sizes => Size, options => TensorOptions);
nif_impl!(rand, Tensor, sizes => Size, options => TensorOptions);
nif_impl!(randn, Tensor, sizes => Size, options => TensorOptions);
nif_impl!(randint, Tensor, low => i64, high => i64, sizes => Size, options => TensorOptions);
nif_impl!(full, Tensor, sizes => Size, scalar => Scalar, options => TensorOptions);
nif_impl!(eye, Tensor, n => i64, m => i64, options => TensorOptions);
nif_impl!(arange, Tensor, start => Scalar, end => Scalar, step => Scalar, options => TensorOptions);
nif_impl!(linspace, Tensor, start => Scalar, end => Scalar, steps => i64, options => TensorOptions);
nif_impl!(logspace, Tensor, start => Scalar, end => Scalar, steps => i64, base => Scalar, options => TensorOptions);
nif_impl!(tensor, Tensor, list => ListWrapper, options => TensorOptions);
