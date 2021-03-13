extern crate rustler;
use rustler::resource::ResourceArc;
use rustler::{Env, Error, Term};
use rustler_sys::enif_make_ref;

use cxx::SharedPtr;

use crate::native::torch;
use crate::rustler::Encoder;
use crate::shared_types::{ListWrapper, TensorStruct};
use crate::conversion::info_types::{unpack_size, wrap_size, wrap_device, wrap_str_atom};
use crate::conversion::scalar_types::unpack_scalar_typed;


pub fn unpack_scalar_list<'a>(list: Vec<Term<'a>>, dtype: String) -> Result<Vec<torch::Scalar>, Error> {
    let mut scalar_list: Vec<torch::Scalar> = Vec::new();
    for term in list {
        // let scalar = find_scalar(term)?;
        let scalar = unpack_scalar_typed(term, &dtype)?;
        scalar_list.push(scalar);
    }
    Ok(scalar_list)
}


pub fn unpack_list_wrapper<'a>(
    index: usize,
    _env: Env<'a>,
    args: &[Term<'a>],
) -> Result<torch::ScalarList, Error> {
    let list_wrapper: ListWrapper = args[index].decode()?;
    let tensor_size = unpack_size(list_wrapper.size)?;
    let coerced_type: String = list_wrapper.dtype.atom_to_string()?;
    let list_scalar = unpack_scalar_list(list_wrapper.list, coerced_type)?;
    let scalar_list = torch::ScalarList {
        list: list_scalar,
        size: tensor_size,
    };
    Ok(scalar_list)
}


pub fn wrap_tensor<'a>(
    tensor_ref: Result<SharedPtr<torch::CrossTensor>, cxx::Exception>,
    env: Env<'a>,
) -> Result<Term<'a>, Error> {
    match tensor_ref {
        Ok(result) => {
            let size = wrap_size(env, torch::size(&result))?; // size_ref(env, &result)?;
            let dtype = wrap_str_atom(env, torch::dtype(&result))?;
            let device = wrap_device(env, torch::device(&result))?;
            let cross_tensor_ref = torch::CrossTensorRef { tensor: result };
            let resource = ResourceArc::new(cross_tensor_ref);
            let reference = unsafe { Term::new(env, enif_make_ref(env.as_c_arg())) };
            let tensor_struct = TensorStruct {
                resource: resource,
                reference: reference,
                size: size,
                dtype: dtype,
                device: device,
            };
            Ok(tensor_struct.encode(env))
        }

        Err(err) => {
            let err_msg = err.what();
            let err_str = err_msg.to_owned();
            let err_parts: Vec<&str> = err_str.split("\n").collect();
            let main_msg = err_parts[0].to_owned();
            Err(Error::RaiseTerm(Box::new(main_msg)))
        }
    }
}
