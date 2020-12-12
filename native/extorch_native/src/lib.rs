#[macro_use]
extern crate rustler;

use rustler::{Encoder, Env, Error, Term};

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
    extern "Rust" {
        // type MultiBuf;

        // fn next_chunk(dims: ) -> &[u8];
        // fn create_int_array_ref(dims: array[i32]) -> &fuIntArrayRef;
    }

    unsafe extern "C++" {
        include!("extorch_native/include/wrapper.h");
        fn test_cpp();
    }

    // unsafe extern "C++" {
    //     include!("torch/torch.h");

    //     type Tensor;
    //     type IntArrayRef;
    //     type TensorOptions;
    //     type optional;
    //     type DimnameList;
    //     // type BlobstoreClient;
    //     fn empty<'a>(size: &IntArrayRef, options: &TensorOptions, optional: &optional) -> &'a Tensor;
    //     // fn new_blobstore_client() -> UniquePtr<BlobstoreClient>;
    // }
}



rustler::rustler_export_nifs! {
    "Elixir.ExTorch.Native",
    [
        ("add", 2, add)
    ],
    None
}

fn add<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let num1: i64 = args[0].decode()?;
    let num2: i64 = args[1].decode()?;

    torch::test_cpp();
    // torch::empty([2, 3, 4], -1, -1);
    Ok((atoms::ok(), num1 + num2).encode(env))
}
