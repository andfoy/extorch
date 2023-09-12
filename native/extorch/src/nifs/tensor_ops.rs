use crate::native::torch;
use crate::shared_types::{Size, TensorIndex, TensorStruct};

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
