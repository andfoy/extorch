use crate::native::torch;
use crate::shared_types::{AtomString, Size, TensorIndex, TensorStruct};
use crate::torch::{IntListOrInt, Scalar, TensorList, TensorOrInt, TensorOut};

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
    value: TensorStruct<'a>,
    inplace: bool
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

nif_impl!(
    dsplit,
    TensorList,
    input: TensorStruct<'a>,
    indices_or_sections: IntListOrInt);

nif_impl!(
    column_stack,
    TensorStruct<'a>,
    tensors: TensorList,
    out: TensorOut
);

nif_impl!(
    dstack,
    TensorStruct<'a>,
    tensors: TensorList,
    out: TensorOut
);

nif_impl!(
    gather,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: i64,
    index_param: TensorStruct<'a>,
    sparse_grad: bool,
    out: TensorOut
);

nif_impl!(
    hsplit,
    TensorList,
    input: TensorStruct<'a>,
    indices_or_sections: IntListOrInt);

nif_impl!(
    hstack,
    TensorStruct<'a>,
    tensors: TensorList,
    out: TensorOut
);

nif_impl!(
    index_add,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: i64,
    index_param: TensorStruct<'a>,
    source: TensorStruct<'a>,
    alpha: Scalar,
    out: TensorOut,
    inplace: bool
);

nif_impl!(
    index_copy,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: i64,
    index_param: TensorStruct<'a>,
    source: TensorStruct<'a>,
    out: TensorOut,
    inplace: bool
);

nif_impl!(
    index_reduce,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: i64,
    index_param: TensorStruct<'a>,
    source: TensorStruct<'a>,
    reduce: AtomString,
    include_self: bool,
    out: TensorOut,
    inplace: bool
);

nif_impl!(
    index_select,
    TensorStruct<'a>,
    input: TensorStruct<'a>,
    dim: i64,
    index_param: TensorStruct<'a>,
    out: TensorOut
);
