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

    struct CrossTensorRef {
        tensor: SharedPtr<CrossTensor>,
    }

    extern "Rust" {
        // type MultiBuf;

        // fn next_chunk(dims: ) -> &[u8];
        // fn create_int_array_ref(dims: array[i32]) -> &fuIntArrayRef;
    }

    unsafe extern "C++" {
        include!("extorch_native/include/wrapper.h");

        type CrossTensor;
        fn empty(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: String,
            requires_grad: bool,
            pin_memory: bool,
        ) -> SharedPtr<CrossTensor>;
        fn size(tensor: &SharedPtr<CrossTensor>) -> &'static [i64];
        fn dtype(tensor: &SharedPtr<CrossTensor>) -> String;
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
        ("empty", 6, empty),
        ("dtype", 1, dtype)
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
    let ex_device: String = args[3].atom_to_string()?;
    let requires_grad: bool = args[4].decode()?;
    let pin_memory: bool = args[5].decode()?;

    let mut sizes: Vec<i64> = Vec::new();
    for term in ex_sizes {
        sizes.push(term.decode()?);
    }

    let tensor_ref = torch::empty(sizes,
                                  ex_dtype,
                                  ex_layout,
                                  ex_device,
                                  requires_grad, pin_memory);
    let cross_tensor_ref = torch::CrossTensorRef { tensor: tensor_ref };
    let resource = ResourceArc::new(cross_tensor_ref);
    Ok(resource.encode(env))
}

fn size<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let resource: ResourceArc<torch::CrossTensorRef> = args[0].decode()?;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;
    let sizes = torch::size(tensor_ref);
    let enc_sizes: Vec<Term<'a>> = sizes.iter().map(|size| size.encode(env)).collect();
    let tuple_sizes = make_tuple(env, &enc_sizes);
    Ok((atoms::ok(), tuple_sizes).encode(env))
}

fn dtype<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let resource: ResourceArc<torch::CrossTensorRef> = args[0].decode()?;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;
    let dtype = torch::dtype(tensor_ref);
    let atom_dtype = Atom::from_str(env, &dtype)?;
    Ok((atoms::ok(), atom_dtype).encode(env))
}
