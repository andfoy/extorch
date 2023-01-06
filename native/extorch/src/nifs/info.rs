
use crate::native::torch;
use crate::shared_types::{TensorOptions, TensorStruct, Reference, AtomString, Size};

use rustler::{Atom, Env, Error, Term, ResourceArc, NifResult};

#[rustler::nif]
fn repr<'a>(tensor: TensorStruct<'a>) -> NifResult<String> {
    let resource = tensor.resource;
    let out = torch::repr(&resource.tensor);
    Ok(out)
}
// nif_impl!(size, Size, tensor => Tensor);
// nif_impl!(device, Device, tensor => Tensor);
// nif_impl!(dtype, StrAtom, tensor => Tensor);
// nif_impl!(repr, String, tensor => Tensor);
// nif_impl!(to_list, ListWrapper, tensor => Tensor);
