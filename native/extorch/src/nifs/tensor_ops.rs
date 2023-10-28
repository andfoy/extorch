use crate::native::torch;
use crate::shared_types::{Size, TensorIndex, TensorStruct};
use crate::torch::{TensorList, TensorOrInt, TensorOut};

use rustler::{Error, NifResult};

nif_impl!(
    unsqueeze,
    TensorStruct<'a>,
    tensor: TensorStruct<'a>,
    dim: i64
);

nif_impl!(
    reshape,
    TensorStruct<'a>,
    tensor: TensorStruct<'a>,
    shape: Size
);

nif_impl!(
    index,
    TensorStruct<'a>,
    tensor: TensorStruct<'a>,
    indices: TensorIndex
);
nif_impl!(
    index_put,
    TensorStruct<'a>,
    tensor: TensorStruct<'a>,
    indices: TensorIndex,
    value: TensorStruct<'a>
);

nif_impl!(conj, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(adjoint, TensorStruct<'a>, input: TensorStruct<'a>);
nif_impl!(transpose, TensorStruct<'a>, input: TensorStruct<'a>, dim0: i64, dim1: i64);
nif_impl!(cat, TensorStruct<'a>, seq: TensorList, dim: i64, out: TensorOut);
nif_impl!(chunk, TensorList, input: TensorStruct<'a>, chunks: i64, dim: i64);
nif_impl!(
    tensor_split,
    TensorList,
    input: TensorStruct<'a>,
    indices_or_sections: TensorOrInt,
    dim: i64);
