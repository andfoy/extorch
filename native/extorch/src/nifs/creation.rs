// use crate::conversion::{
//     unpack_list_wrapper, unpack_scalar, unpack_size_init, unpack_tensor_options, wrap_tensor,
// };
use crate::native::torch;
use crate::shared_types::{TensorOptions, TensorStruct, Reference, AtomString, Size};

use rustler::{Atom, Env, Error, Term, ResourceArc, NifResult};


// #[rustler::nif]
// pub fn empty<'a>(size: Size, options: TensorOptions) -> NifResult<TensorStruct<'a>> {
//     let unwrapped_size = size.size;
//     let tensor_or_err = torch::empty(
//         unwrapped_size, options.dtype.name, options.layout.name, options.device,
//         options.requires_grad, options.pin_memory,
//         options.memory_format.name);

//     match tensor_or_err {
//         Ok(shared_tensor) => {
//             let size = Size{ size: torch::size(&shared_tensor).to_vec() };
//             let device = torch::device(&shared_tensor);
//             let dtype_name: String = torch::dtype(&shared_tensor);
//             let dtype = AtomString { name: dtype_name };

//             let wrapped_tensor = torch::CrossTensorRef {
//                 tensor: shared_tensor
//             };

//             let resource = ResourceArc::new(wrapped_tensor);
//             let reference = Reference::new();
//             Ok(TensorStruct {
//                 resource,
//                 reference,
//                 size,
//                 dtype,
//                 device
//             })
//         },
//         Err(err) => {
//             let err_msg = err.what();
//             let err_str = err_msg.to_owned();
//             let err_parts: Vec<&str> = err_str.split("\n").collect();
//             let main_msg = err_parts[0].to_owned();
//             Err(Error::RaiseTerm(Box::new(main_msg)))
//         }
//     }
// }


// trace_macros!(true);
nif_impl!(empty, TensorStruct<'a>, size: Size, options: TensorOptions);
// trace_macros!(false);

// nif_impl!(empty, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(zeros, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(ones, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(rand, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(randn, Tensor, sizes => Size, options => TensorOptions);
// nif_impl!(randint, Tensor, low => i64, high => i64, sizes => Size, options => TensorOptions);
// nif_impl!(full, Tensor, sizes => Size, scalar => Scalar, options => TensorOptions);
// nif_impl!(eye, Tensor, n => i64, m => i64, options => TensorOptions);
// nif_impl!(arange, Tensor, start => Scalar, end => Scalar, step => Scalar, options => TensorOptions);
// nif_impl!(linspace, Tensor, start => Scalar, end => Scalar, steps => i64, options => TensorOptions);
// nif_impl!(logspace, Tensor, start => Scalar, end => Scalar, steps => i64, base => Scalar, options => TensorOptions);
// nif_impl!(tensor, Tensor, list => ListWrapper, options => TensorOptions);
