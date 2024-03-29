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
use crate::nifs::*;
pub use crate::shared_types::TensorStruct;
use rustler::{Env, Term};

fn load(env: Env, _: Term) -> bool {
    rustler::resource!(torch::CrossTensorRef, env);
    true
}

#[rustler::nif]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

rustler::init!(
    "Elixir.ExTorch.Native",
    [
        add,

        // Tensor information
        repr,
        size,
        dim,
        device,
        dtype,
        requires_grad,
        numel,
        to_list,
        item,
        real,
        imag,
        layout,
        memory_format,
        is_complex,
        is_floating_point,
        is_conj,
        is_nonzero,
        to,

        // Tensor creation
        empty,
        zeros,
        ones,
        rand,
        randn,
        randint,
        full,
        eye,
        arange,
        linspace,
        logspace,
        tensor,
        complex,
        polar,
        view_as_complex,
        ones_like,
        zeros_like,
        empty_like,
        full_like,
        rand_like,
        randn_like,
        randint_like,

        // Tensor manipulation
        unsqueeze,
        reshape,
        index,
        index_put,
        conj,
        adjoint,
        transpose,
        cat,
        chunk,
        tensor_split,
        dsplit,
        column_stack,
        dstack,
        gather,
        hsplit,
        hstack,
        index_add,
        index_copy,
        index_reduce,
        index_select,
        masked_select,
        movedim,
        narrow,
        narrow_copy,
        nonzero,
        permute,
        vstack,
        select,
        scatter,
        diagonal_scatter,
        select_scatter,
        slice_scatter,
        scatter_add,
        scatter_reduce,
        split,
        squeeze,
        stack,
        t,
        take,
        take_along_dim,

        // Tensor comparing operations
        allclose,
        argsort,
        sort,
        eq,
        equal,
        ge,
        gt,
        le,
        lt,
        ne,
        isclose,
        isfinite,
        isinf,
        isposinf,
        isneginf,
        isnan,
        isreal,
        isin,
        kthvalue,
        maximum,
        minimum,
        fmax,
        fmin,
        topk,
        msort,

        // Tensor reduction operations
        argmax,
        argmin,
        all,
        any,
        max,
        min,
        amax,
        amin,
        aminmax,
        dist,
        logsumexp,
        sum,
        nansum,
        mean,
        nanmean,
        median,
        nanmedian,
        mode,
        prod,
        quantile,
        nanquantile,
        std_dev,
        std_mean,
        unique,
        unique_consecutive,
        var,
        var_mean,
        count_nonzero,

        // Other operations
        resolve_conj,
    ],
    load = load
);
