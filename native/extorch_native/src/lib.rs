//! [Github](https://github.com/andfoy/extorch)
//!
//! ExTorch-Native is a library that wraps C++ libtorch functions into Erlang NIFs.
//!
//! As well as providing safe NIFs, this crate also offers the opportunity to extend the
//! ExTorch ecosystem by providing access also to the original C++ libtorch namespace
//! via the [`crate::torch`] module, which is implemented using
//! [`cxx::bridge`][cxx::bridge].
//!
//! Additionally, this library provides access to the [`nif_impl`] macro, which allows
//! to autogenerate a NIF wrapper to a libtorch/ecosystem function call.
//!
//! The library provides functionality for both Erlang and Elixir, however Elixir is favored as of
//! now.

#[macro_use]
extern crate rustler;

// Local imports
#[macro_use]
mod macros;
mod conversion;
mod native;
mod nifs;
mod shared_types;

pub use crate::native::torch;#[doc(hidden)]
use crate::nifs::*;
use rustler::{Encoder, Env, Error, Term};

// RUSTFLAGS="-C link-args=-Wl,-rpath,/home/andfoy/anaconda3/lib/python3.7/site-packages/torch/lib"
mod atoms {
    rustler_atoms! {
        atom ok;
        //atom error;
        //atom __true__ = "true";
        //atom __false__ = "false";
    }
}

rustler::rustler_export_nifs! {
    "Elixir.ExTorch.Native",
    [
        ("add", 2, add),
        ("empty", 7, empty),
        ("zeros", 7, zeros),
        ("ones", 7, ones),
        ("rand", 7, rand),
        ("randn", 7, randn),
        ("randint", 9, randint),
        ("full", 8, full),
        ("eye", 8, eye),
        ("arange", 9, arange),
        ("linspace", 9, linspace),
        ("logspace", 10, logspace),
        ("tensor", 7, tensor),
        ("repr", 1, repr),
        ("size", 1, size),
        ("dtype", 1, dtype),
        ("device", 1, device),
        ("unsqueeze", 2, unsqueeze)
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
