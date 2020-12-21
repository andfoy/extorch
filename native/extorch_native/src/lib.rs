#[macro_use]
extern crate rustler;

// use cxx::SharedPtr;
use rustler::resource::ResourceArc;
// use rustler::types::list::ListIterator;
use rustler::types::Atom;
use rustler::types::tuple::{get_tuple, make_tuple};
use rustler::{Encoder, Env, Error, Term};
// use std::ptr::NonNull;

// RUSTFLAGS="-C link-args=-Wl,-rpath,/home/andfoy/anaconda3/lib/python3.7/site-packages/torch/lib"
mod atoms {
    rustler_atoms! {
        atom ok;
        //atom error;
        //atom __true__ = "true";
        //atom __false__ = "false";
    }
}

#[cxx::bridge]
mod torch {
    /// Shared interface to a tensor pointer in memory.
    struct CrossTensorRef {
        tensor: SharedPtr<CrossTensor>,
    }

    /// Torch tensor device descriptor.
    struct Device {
        device: String,
        index: i64
    }

    extern "Rust" {

    }

    unsafe extern "C++" {
        include!("extorch_native/include/wrapper.h");

        /// Reference to a torch tensor in memory
        type CrossTensor;

        /// Get the size of a tensor
        fn size(tensor: &SharedPtr<CrossTensor>) -> &'static [i64];
        /// Get the type of a tensor
        fn dtype(tensor: &SharedPtr<CrossTensor>) -> String;
        /// Get the device where the tensor lives on
        fn device(tensor: &SharedPtr<CrossTensor>) -> Device;

        fn empty(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String
        ) -> Result<SharedPtr<CrossTensor>>;
    }
}

// struct Wrapper(NonNull<torch::CrossTensorRef>);
unsafe impl std::marker::Send for torch::CrossTensorRef {}
unsafe impl std::marker::Sync for torch::CrossTensorRef {}

rustler::rustler_export_nifs! {
    "Elixir.ExTorch.Native",
    [
        ("add", 2, add),
        ("size", 1, size),
        ("empty", 7, empty),
        ("dtype", 1, dtype),
        ("device", 1, device)
    ],
    Some(on_load)
}

fn on_load(env: Env, _info: Term) -> bool {
    resource_struct_init!(torch::CrossTensorRef, env);
    true
}

fn add<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let num1: i64 = args[0].decode()?;
    let num2: i64 = args[1].decode()?;

    Ok((atoms::ok(), num1 + num2).encode(env))
}

fn empty<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let tuple_sizes: Term<'a> = args[0];
    let ex_sizes: Vec<Term<'a>>;

    match get_tuple(tuple_sizes) {
        Ok(sizes) => {
            ex_sizes = sizes;
        }
        Err(_err) => {
            ex_sizes = tuple_sizes.decode()?;
        }
    }

    let ex_dtype: String = args[1].atom_to_string()?;
    let ex_layout: String = args[2].atom_to_string()?;
    // let ex_device: String = args[3].atom_to_string()?;
    let tuple_device: Term<'a> = args[3];

    let mut device_name;
    let mut device_index: i64;

    match get_tuple(tuple_device) {
        Ok(device_tuple) => {
            device_name = device_tuple[0].atom_to_string()?;
            device_index = device_tuple[1].decode()?;
        }

        Err(_err) => {
            let ex_device: String;
            match args[3].atom_to_string() {
                Ok(dev) => {
                    ex_device = dev;
                }
                Err(_err) => {
                    ex_device = args[3].decode()?;
                }
            }
            // let ex_device: String = ?;
            device_name = ex_device.clone();
            device_index = -1;

            let device_copy = &ex_device;
            if device_copy.contains(":") {
                let device_parts: Vec<&str> = device_copy.split(":").collect();
                device_name = device_parts[0].to_owned();
                device_index = device_parts[1].parse().unwrap();
            }
        }
    }

    let requires_grad: bool = args[4].decode()?;
    let pin_memory: bool = args[5].decode()?;
    let memory_format: String = args[6].atom_to_string()?;

    let mut sizes: Vec<i64> = Vec::new();
    for term in ex_sizes {
        sizes.push(term.decode()?);
    }

    let device = torch::Device {index: device_index, device: device_name};
    let tensor_ref = torch::empty(sizes,
                                  ex_dtype,
                                  ex_layout,
                                  device,
                                  requires_grad,
                                  pin_memory,
                                  memory_format);
    match tensor_ref {
        Ok(result) => {
            let cross_tensor_ref = torch::CrossTensorRef { tensor: result };
            let resource = ResourceArc::new(cross_tensor_ref);
            Ok(resource.encode(env))
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

fn size<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let resource: ResourceArc<torch::CrossTensorRef> = args[0].decode()?;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;
    let sizes = torch::size(tensor_ref);
    let enc_sizes: Vec<Term<'a>> = sizes.iter().map(|size| size.encode(env)).collect();
    let tuple_sizes = make_tuple(env, &enc_sizes);
    Ok(tuple_sizes.encode(env))
}

fn dtype<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let resource: ResourceArc<torch::CrossTensorRef> = args[0].decode()?;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;
    let dtype = torch::dtype(tensor_ref);
    let atom_dtype = Atom::from_str(env, &dtype)?;
    Ok(atom_dtype.encode(env))
}

fn device<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let resource: ResourceArc<torch::CrossTensorRef> = args[0].decode()?;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;

    let device = torch::device(tensor_ref);
    let device_name = device.device;
    let device_index = device.index;

    let atom_device = Atom::from_str(env, &device_name)?;
    let mut return_term: Term<'a> = atom_device.encode(env);
    if device_index != -1 {
        let enc_index = device_index.encode(env);
        return_term = make_tuple(env, &[return_term, enc_index]);
    }
    Ok(return_term.encode(env))
}
