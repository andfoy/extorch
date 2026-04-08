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
//!

// #![feature(trace_macros)]
#[macro_use]
mod macros;
mod encoding;
mod native;
mod nifs;
mod shared_types;

#[doc(hidden)]
pub use crate::native::torch;
pub use crate::shared_types::TensorStruct;
use rustler::{Env, Term};

impl rustler::Resource for torch::CrossTensorRef {}
impl rustler::Resource for torch::CrossModuleRef {}
impl rustler::Resource for torch::CrossNNModuleRef {}

fn load(env: Env, _: Term) -> bool {
    env.register::<torch::CrossTensorRef>().is_ok()
        && env.register::<torch::CrossModuleRef>().is_ok()
        && env.register::<torch::CrossNNModuleRef>().is_ok()
}

#[rustler::nif]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

rustler::init!("Elixir.ExTorch.Native", load = load);
