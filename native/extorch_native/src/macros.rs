
#[macro_export]
macro_rules! call (
    ($f: path, $y:ident => $yt:ident) => {
        make_call!($f, (), $y => $yt)
    };
    ($f: path, $($y:ident => $yt:ident),+) => {
        make_call!($f, (), $($y => $yt),+)
    };
);

#[macro_export]
macro_rules! make_call {
    ($f: path, ($($args:expr)*)) => { $f($($args),*) };
    ($f: path, ($($args:expr)*), $x:ident => TensorOptions) => {
        make_call!($f, ($($args)* $x.dtype $x.layout $x.device $x.requires_grad $x.pin_memory $x.memory_format))
    };
    ($f: path, (), $x:ident => TensorOptions, $($y:ident => $yt:ident),+) => {
        make_call!($f, ($x.dtype $x.layout $x.device $x.requires_grad $x.pin_memory $x.memory_format), $($y => $yt),+)
    };
    ($f: path, ($($args:expr)*), $x:ident => TensorOptions, $($y:ident => $yt:ident),+) => {
        make_call!($f, ($($args)* $x.dtype $x.layout $x.device $x.requires_grad $x.pin_memory $x.memory_format), $($y => $yt),+)
    };
    ($f: path, ($($args:expr)*), $x:ident => $tt:ident) => {
        make_call!($f, ($($args)* $x))
    };
    ($f: path, (), $x:ident => $tt:ident, $($y:ident => $yt:ident),+) => {
        make_call!($f, ($x), $($y => $yt),+)
    };
    ($f: path, ($($args:expr)*), $x:ident => $tt:ident, $($y:ident => $yt:ident),+) => {
        make_call!($f, ($($args)* $x), $($y => $yt),+)
    };
}

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
    ($result:ident, $ret_type:ident, $env:ident, $args:ident) => {
        // wrap_device($env, $result)
        Ok($result.encode($env))
    };
}

#[macro_export]
macro_rules! nif_impl {
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
}
