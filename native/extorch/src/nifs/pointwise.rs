use crate::native::torch;
use crate::shared_types::{Size, TensorStruct};
use crate::torch::Scalar;

use rustler::{Error, NifResult};

nif_impl!(real, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(imag, TensorStruct<'a>, input: TensorStruct<'a>);

// Arithmetic
nif_impl!(add, TensorStruct<'a>, input: TensorStruct<'a>, other: TensorStruct<'a>, alpha: Scalar);
nif_impl!(sub, TensorStruct<'a>, input: TensorStruct<'a>, other: TensorStruct<'a>, alpha: Scalar);
nif_impl!(mul, TensorStruct<'a>, input: TensorStruct<'a>, other: TensorStruct<'a>);
nif_impl!(tensor_div, TensorStruct<'a>, input: TensorStruct<'a>, other: TensorStruct<'a>);
nif_impl!(neg, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(tensor_abs, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(pow_tensor, TensorStruct<'a>, input: TensorStruct<'a>, exponent: Scalar);

nif_impl!(clamp, TensorStruct<'a>, input: TensorStruct<'a>, min_val: Scalar, max_val: Scalar);

// Math
nif_impl!(tensor_exp, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(tensor_log, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(tensor_sqrt, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(tensor_sin, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(tensor_cos, TensorStruct<'a>, input: TensorStruct<'a>);

// Linear algebra
nif_impl!(matmul, TensorStruct<'a>, input: TensorStruct<'a>, other: TensorStruct<'a>);
nif_impl!(mm, TensorStruct<'a>, input: TensorStruct<'a>, other: TensorStruct<'a>);
nif_impl!(bmm, TensorStruct<'a>, input: TensorStruct<'a>, other: TensorStruct<'a>);

// Conditional / masking
nif_impl!(tensor_where, TensorStruct<'a>, condition: TensorStruct<'a>, x: TensorStruct<'a>, y: TensorStruct<'a>);
nif_impl!(masked_fill, TensorStruct<'a>, input: TensorStruct<'a>, mask: TensorStruct<'a>, value: Scalar);

// Tensor manipulation
nif_impl!(contiguous, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(clone, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(detach, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(view, TensorStruct<'a>, input: TensorStruct<'a>, shape: Size);
nif_impl!(expand, TensorStruct<'a>, input: TensorStruct<'a>, shape: Size);

// Functional activations
nif_impl!(functional_softmax, TensorStruct<'a>, input: TensorStruct<'a>, dim: i64);
nif_impl!(functional_log_softmax, TensorStruct<'a>, input: TensorStruct<'a>, dim: i64);
nif_impl!(functional_relu, TensorStruct<'a>, input: TensorStruct<'a>);

// Einsum
nif_impl!(einsum, TensorStruct<'a>, equation: String, a: TensorStruct<'a>, b: TensorStruct<'a>);
