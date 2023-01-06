#[doc(hidden)]
#[macro_export]
macro_rules! call {
    ($f: path $(, $y:ident : $yt:tt)*) => {
        make_call!($f, (), ($($y : $yt),*))
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! make_call {
    ($f: path, ($($args:expr)*), ()) => { $f($($args),*) };
    ($f: path, ($($args:expr)*), ($x:ident : Size $(, $argsf:ident : $argst:tt)*)) => {
        make_call!($f, ($($args)* $x.size), ($($argsf : $argst),*))
    };
    ($f: path, ($($args:expr)*), ($x:ident : TensorOptions $(, $argsf:ident : $argst:tt)*)) => {
        make_call!($f, ($($args)* $x.dtype.name $x.layout.name $x.device $x.requires_grad
                        $x.pin_memory $x.memory_format.name), ($($argsf : $argst),*))
    };
    ($f: path, ($($args:expr)*), ($x:ident : $y:tt $(, $argsf:ident : $argst:tt)*)) => {
        make_call!($f, ($($args)* $x), ($($argsf : $argst),*))
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! unpack_arg {
    ($pos:ident, $env:ident, $args:ident, $x:ident, Size) => {
        let $x = unpack_size_init($pos, $env, $args)?;
    };
    ($pos:ident, $env:ident, $args:ident, $x:ident, Scalar) => {
        let $x = unpack_scalar($pos, $env, $args)?;
    };
    ($pos:ident, $env:ident, $args:ident, $x:ident, TensorOptions) => {
        let $x = unpack_tensor_options($pos, $env, $args)?;
    };
    ($pos:ident, $env:ident, $args:ident, $x:ident, ListWrapper) => {
        let $x = unpack_list_wrapper($pos, $env, $args)?;
    };
    ($pos:ident, $env:ident, $args:ident, $x:ident, Tensor) => {
        let $x = &(&*(($args[$pos].decode::<TensorStruct>()?).resource)).tensor;
    };
    ($pos:ident, $env:ident, $args:ident, $x:ident, $tt:ident) => {
        let $x: $tt = $args[$pos].decode()?;
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! unpack_args {
    // Base case:
    ($pos:ident, $env:ident, $args:ident, $x:ident => $tt:ident) => {
        unpack_arg!($pos, $env, $args, $x, $tt);
    };
    // `$x` followed by at least one `$y,`
    ($pos:ident, $env:ident, $args:ident, $x:ident => $tt:ident, $($y:ident => $yt:ident),+) => {
        unpack_arg!($pos, $env, $args, $x, $tt);
        $pos += 1;
        unpack_args!($pos, $env, $args, $($y => $yt),+);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! wrap_result {
    ($result:ident, Tensor, $env:ident, $args:ident) => {
        wrap_tensor($result, $env)
    };
    ($result:ident, Device, $env:ident, $args:ident) => {
        wrap_device($env, $result)
    };
    ($result:ident, Size, $env:ident, $args:ident) => {
        wrap_size($env, $result)
    };
    ($result:ident, StrAtom, $env:ident, $args:ident) => {
        wrap_str_atom($env, $result)
    };
    ($result:ident, ListWrapper, $env:ident, $args:ident) => {
        wrap_list_wrapper($env, $result)
    };
    ($result:ident, $ret_type:ident, $env:ident, $args:ident) => {
        // wrap_device($env, $result)
        Ok($result.encode($env))
    };
}

/// Define torch NIF automatically given a function name defined in [`crate::torch`]
/// (or other [`cxx::bridge`] module that interfaces with libtorch), the return type,
/// and a list of arguments.
///
/// For example:
///
/// ```
/// // This exports the torch::ones function as a NIF
/// nif_impl!(ones, Tensor, sizes: Size, options: TensorOptions);
/// ```
///
/// # Arguments
/// * `func_name`: the name of the function as it is defined in the [`crate::torch`] module.
/// * `return_type`: the return type of the function, as defined below.
/// * `call_path`: optional, the path to the [`cxx::bridge`] that declares the function. Defaults to [`crate::torch`]
/// * `arg_name => arg_type`: the name of the argument to pass to the function and its type (defined below)
///
/// # Input Types
/// * `Tensor`: If the input is a Tensor.
/// * `Size`: If the input is a tuple/list of integer numbers.
/// * `Scalar`: If the input is an integer/floating type number of any precision.
/// * `TensorOptions`: If the input refers to a struct of tensor initialization options.
/// * `ListWrapper`: If the input is a list with numbers.
///
/// # Return Types
/// * `Tensor`: If the output is a Tensor.
/// * `Device`: If the output corresponds to the device of a Tensor.
/// * `Size`: If the output corresponds to a tuple of integer numbers.
/// * `StrAtom`: If the output corresponds to an atom built from a string.
///
/// **Note:** This macro also accepts Rust base types, like [`i32`], [`bool`], [`i64`],
/// [`String`], etc. For both input and output types.
///
#[macro_export]
macro_rules! nif_impl {
    ($func_name:ident, $ret_type:ty $(, $argsf:ident : $argst:tt)*) => {
        #[rustler::nif]
        pub fn $func_name<'a>($($argsf: $argst),*) -> NifResult<$ret_type> {
            // unpack_args!($($argsf: $argst),*)
            let result = call!(torch::$func_name $(, $argsf: $argst)*);
            match result {
                Ok(res) => {
                    let result_wrapped = Into::<$ret_type>::into(res);
                    Ok(result_wrapped)
                },
                Err(err) => {
                    let err_msg = err.what();
                    let err_str = err_msg.to_owned();
                    let err_parts: Vec<&str> = err_str.split("\n").collect();
                    let main_msg = err_parts[0].to_owned();
                    Err(Error::RaiseTerm(Box::new(main_msg)))
                }
            }

        }
    };
    ($func_name:ident, $ret_type:ident, $($argsf:ident => $argst:ident),+) => {
        #[allow(unused_mut)]
        pub fn $func_name<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
            let mut pos: usize = 0;
            unpack_args!(pos, env, args, $($argsf => $argst),+);
            let result = call!(torch::$func_name, $($argsf => $argst),+);
            // wrap_tensor(tensor_ref, env, args)
            wrap_result!(result, $ret_type, env, args)
        }
    };
    ($func_name:ident, $ret_type:ident, $call_path:path, $($argsf:ident => $argst:ident),+) => {
        #[allow(unused_mut)]
        pub fn $func_name<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
            let mut pos: usize = 0;
            unpack_args!(pos, env, args, $($argsf => $argst),+);
            let result = call!($call_path::$func_name, $($argsf => $argst),+);
            // wrap_tensor(tensor_ref, env, args)
            wrap_result!(result, $ret_type, env, args)
        }
    };
}
