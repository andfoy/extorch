#[macro_use]
extern crate rustler;

use lazy_static::lazy_static;
use std::collections::HashMap;

use cxx::SharedPtr;
use rustler::resource::ResourceArc;
// use rustler::types::list::ListIterator;
use rustler::types::tuple::{get_tuple, make_tuple};
use rustler::types::Atom;
use rustler::{Encoder, Env, Error, Term};
use rustler_sys::enif_make_ref;
// use std::ptr::NonNull;

lazy_static! {
    static ref ALL_TYPES: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("uint8", "uint8");
        m.insert("int8", "int8");
        m.insert("int16", "int16");
        m.insert("int32", "int32");
        m.insert("int64", "int64");
        m.insert("float16", "float16");
        m.insert("float32", "float32");
        m.insert("float64", "float64");
        m.insert("bfloat16", "bfloat16");
        m.insert("byte", "uint8");
        m.insert("char", "int8");
        m.insert("short", "int16");
        m.insert("int", "int32");
        m.insert("long", "int64");
        m.insert("half", "float16");
        m.insert("float", "float32");
        m.insert("double", "float64");
        m.insert("bool", "bool");
//     {"complex32", torch::kComplexHalf},
//     {"complex64", torch::kComplexFloat},
//     {"complex128", torch::kComplexDouble},
//     {"bool", torch::kBool}
        m
    };
}

// RUSTFLAGS="-C link-args=-Wl,-rpath,/home/andfoy/anaconda3/lib/python3.7/site-packages/torch/lib"
mod atoms {
    rustler_atoms! {
        atom ok;
        //atom error;
        //atom __true__ = "true";
        //atom __false__ = "false";
    }
}

// macro_rules! unpack_arg {
//     ($pos:literal, $arg:ident:Tensor) => {
//         wrapper_$arg: TensorStruct = args[$pos].decode()?;
//         // let wrapper: TensorStruct = args[0].decode()?;
//         let resource_$arg = wrapper_$arg.resource;
//         // let resource: ResourceArc<torch::CrossTensorRef> = args[0].decode()?;
//         let cross_tensor_ref_$arg = &*resource_$arg;
//         let $arg = &cross_tensor_ref_$arg.tensor;
//     };
// }

// macro_rules! unpack_args {
//     ($arg:expr) => {
//         unpack_arg!($arg)
//     }
//     ($arg:expr, $(args:expr),+) => {
//         unpack_arg!($arg)
//         unpack_args!($($args),+)
//     };
// }

// macro_rules! nif_impl {
//     ($func_name:ident, Tensor, $args:expr) => {
//         fn $func_name<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
//             unpack_args!($args)
//             tensor_ref = torch::$func_name(
//                 args_without_type!($args)
//             )
//             wrap_tensor(tensor_ref, env, args)
//         }
//     };
// }

#[cxx::bridge]
mod torch {
    /// Shared interface to a tensor pointer in memory.
    struct CrossTensorRef {
        tensor: SharedPtr<CrossTensor>,
    }

    /// Torch tensor device descriptor.
    struct Device {
        device: String,
        index: i64,
    }

    /// Scalar number representation
    struct Scalar {
        _ui8: u8,
        _i8: i8,
        _i16: i16,
        _i32: i32,
        _i64: i64,
        _f16: f32,
        _f32: f32,
        _f64: f64,
        _bool: bool,
        entry_used: String,
    }

    extern "Rust" {}

    unsafe extern "C++" {
        include!("extorch_native/include/wrapper.h");

        /// Reference to a torch tensor in memory
        type CrossTensor;

        // Tensor attribute access
        /// Get the size of a tensor
        fn size(tensor: &SharedPtr<CrossTensor>) -> &'static [i64];
        /// Get the type of a tensor
        fn dtype(tensor: &SharedPtr<CrossTensor>) -> String;
        /// Get the device where the tensor lives on
        fn device(tensor: &SharedPtr<CrossTensor>) -> Device;
        /// Get a string representation of a tensor
        fn repr(tensor: &SharedPtr<CrossTensor>) -> String;

        // Tensor creation ops
        /// Create an empty tensor
        fn empty(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        /// Create a tensor filled with zeros
        fn zeros(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        /// Create a tensor filled with ones
        fn ones(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn full(
            dims: Vec<i64>,
            value: Scalar,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn eye(
            n: i64,
            m: i64,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;
    }
}

// struct Wrapper(NonNull<torch::CrossTensorRef>);
unsafe impl std::marker::Send for torch::CrossTensorRef {}
unsafe impl std::marker::Sync for torch::CrossTensorRef {}

struct TensorOptions {
    dtype: String,
    layout: String,
    device: torch::Device,
    requires_grad: bool,
    pin_memory: bool,
    memory_format: String,
}

#[derive(NifStruct)]
#[module = "ExTorch.Tensor"]
struct TensorStruct<'a> {
    resource: ResourceArc<torch::CrossTensorRef>,
    reference: Term<'a>,
    size: Term<'a>,
    dtype: Term<'a>,
    device: Term<'a>,
}

rustler::rustler_export_nifs! {
    "Elixir.ExTorch.Native",
    [
        ("add", 2, add),
        ("empty", 7, empty),
        ("zeros", 7, zeros),
        ("ones", 7, ones),
        ("full", 8, full),
        ("eye", 8, eye),
        ("repr", 1, repr),
        ("size", 1, size),
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

fn size<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let wrapper: TensorStruct = args[0].decode()?;
    let resource = wrapper.resource;
    // let resource: ResourceArc<torch::CrossTensorRef> = args[0].decode()?;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;
    size_ref(env, tensor_ref)
}

fn size_ref<'a>(
    env: Env<'a>,
    tensor_ref: &SharedPtr<torch::CrossTensor>,
) -> Result<Term<'a>, Error> {
    let sizes = torch::size(tensor_ref);
    let enc_sizes: Vec<Term<'a>> = sizes.iter().map(|size| size.encode(env)).collect();
    let tuple_sizes = make_tuple(env, &enc_sizes);
    Ok(tuple_sizes.encode(env))
}

fn dtype<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let wrapper: TensorStruct = args[0].decode()?;
    let resource = wrapper.resource;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;
    dtype_ref(env, tensor_ref)
}

fn dtype_ref<'a>(
    env: Env<'a>,
    tensor_ref: &SharedPtr<torch::CrossTensor>,
) -> Result<Term<'a>, Error> {
    let dtype = torch::dtype(tensor_ref);
    let atom_dtype = Atom::from_str(env, &dtype)?;
    Ok(atom_dtype.encode(env))
}

fn device<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let wrapper: TensorStruct = args[0].decode()?;
    let resource = wrapper.resource;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;
    device_ref(env, tensor_ref)
}

fn device_ref<'a>(
    env: Env<'a>,
    tensor_ref: &SharedPtr<torch::CrossTensor>,
) -> Result<Term<'a>, Error> {
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

fn repr<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let wrapper: TensorStruct = args[0].decode()?;
    let resource = wrapper.resource;
    let cross_tensor_ref = &*resource;
    let tensor_ref = &cross_tensor_ref.tensor;

    let str_repr = torch::repr(tensor_ref);
    Ok(str_repr.encode(env))
}

fn unpack_size_init<'a>(index: usize, _env: Env<'a>, args: &[Term<'a>]) -> Result<Vec<i64>, Error> {
    let tuple_sizes: Term<'a> = args[index];
    let ex_sizes: Vec<Term<'a>>;

    match get_tuple(tuple_sizes) {
        Ok(sizes) => {
            ex_sizes = sizes;
        }
        Err(_err) => {
            ex_sizes = tuple_sizes.decode()?;
        }
    }

    let mut sizes: Vec<i64> = Vec::new();
    for term in ex_sizes {
        sizes.push(term.decode()?);
    }
    Ok(sizes)
}

fn unpack_scalar<'a>(
    index: usize,
    s_type: String,
    _env: Env<'a>,
    args: &[Term<'a>],
) -> Result<torch::Scalar, Error> {
    let ex_scalar: Term<'a> = args[index];
    let mut type_cast: String = "float32".to_owned();
    if ALL_TYPES.contains_key::<str>(&s_type) {
        type_cast = ALL_TYPES.get::<str>(&s_type).unwrap().to_string();
    }

    let scalar_result: torch::Scalar;
    // let ref_cast: &str = &type_cast;
    match &type_cast[..] {
        "uint8" => {
            let cast_value: u8 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: cast_value,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: type_cast,
            }
        }
        "int8" => {
            let cast_value: i8 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: cast_value,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: type_cast,
            }
        }
        "int16" => {
            let cast_value: i16 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: cast_value,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: type_cast,
            }
        }
        "int32" => {
            let cast_value: i32 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: cast_value,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: type_cast,
            }
        }
        "int64" => {
            let cast_value: i64 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: cast_value,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: type_cast,
            }
        }
        "float16" => {
            let cast_value: f32 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: cast_value,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: type_cast,
            }
        }
        "bfloat16" => {
            let cast_value: f32 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: cast_value,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: "float16".to_owned(),
            }
        }
        "float32" => {
            let cast_value: f32 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: cast_value,
                _f64: -1.0,
                _bool: false,
                entry_used: type_cast,
            }
        }
        "float64" => {
            let cast_value: f64 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: cast_value,
                _bool: false,
                entry_used: type_cast,
            }
        }
        "bool" => {
            let cast_value: bool = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: cast_value,
                entry_used: type_cast,
            }
        }
        _ => {
            let cast_value: f32 = ex_scalar.decode()?;
            scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: cast_value,
                _f64: -1.0,
                _bool: false,
                entry_used: "float32".to_owned(),
            }
        }
    }
    Ok(scalar_result)
}

fn unpack_tensor_options<'a>(
    off: usize,
    _env: Env<'a>,
    args: &[Term<'a>],
) -> Result<TensorOptions, Error> {
    let ex_dtype: String = args[off + 1].atom_to_string()?;
    let ex_layout: String = args[off + 2].atom_to_string()?;
    // let ex_device: String = args[3].atom_to_string()?;
    let tuple_device: Term<'a> = args[off + 3];

    let mut device_name;
    let mut device_index: i64;

    match get_tuple(tuple_device) {
        Ok(device_tuple) => {
            device_name = device_tuple[0].atom_to_string()?;
            device_index = device_tuple[1].decode()?;
        }

        Err(_err) => {
            let ex_device: String;
            match args[off + 3].atom_to_string() {
                Ok(dev) => {
                    ex_device = dev;
                }
                Err(_err) => {
                    ex_device = args[off + 3].decode()?;
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

    let requires_grad: bool = args[off + 4].decode()?;
    let pin_memory: bool = args[off + 5].decode()?;
    let memory_format: String = args[off + 6].atom_to_string()?;

    let device = torch::Device {
        index: device_index,
        device: device_name,
    };

    Ok(TensorOptions {
        dtype: ex_dtype,
        layout: ex_layout,
        device: device,
        requires_grad: requires_grad,
        pin_memory: pin_memory,
        memory_format: memory_format,
    })
}

fn wrap_tensor<'a>(
    tensor_ref: Result<SharedPtr<torch::CrossTensor>, cxx::Exception>,
    env: Env<'a>,
    _args: &[Term<'a>],
) -> Result<Term<'a>, Error> {
    match tensor_ref {
        Ok(result) => {
            let size = size_ref(env, &result)?;
            let dtype = dtype_ref(env, &result)?;
            let device = device_ref(env, &result)?;
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

fn empty<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let sizes = unpack_size_init(0, env, args)?;
    let options = unpack_tensor_options(0, env, args)?;
    let tensor_ref = torch::empty(
        sizes,
        options.dtype,
        options.layout,
        options.device,
        options.requires_grad,
        options.pin_memory,
        options.memory_format,
    );

    wrap_tensor(tensor_ref, env, args)
}

fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let sizes = unpack_size_init(0, env, args)?;
    let options = unpack_tensor_options(0, env, args)?;
    let tensor_ref = torch::zeros(
        sizes,
        options.dtype,
        options.layout,
        options.device,
        options.requires_grad,
        options.pin_memory,
        options.memory_format,
    );

    wrap_tensor(tensor_ref, env, args)
}

fn ones<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let sizes = unpack_size_init(0, env, args)?;
    let options = unpack_tensor_options(0, env, args)?;
    let tensor_ref = torch::ones(
        sizes,
        options.dtype,
        options.layout,
        options.device,
        options.requires_grad,
        options.pin_memory,
        options.memory_format,
    );

    wrap_tensor(tensor_ref, env, args)
}

fn full<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let sizes = unpack_size_init(0, env, args)?;
    let options = unpack_tensor_options(1, env, args)?;
    let scalar = unpack_scalar(1, options.dtype.clone(), env, args)?;
    let tensor_ref = torch::full(
        sizes,
        scalar,
        options.dtype,
        options.layout,
        options.device,
        options.requires_grad,
        options.pin_memory,
        options.memory_format,
    );
    wrap_tensor(tensor_ref, env, args)
}

fn eye<'a>(env: Env<'a>, args: &[Term<'a>]) -> Result<Term<'a>, Error> {
    let n: i64 = args[0].decode()?;
    let m: i64 = args[1].decode()?;
    let options = unpack_tensor_options(1, env, args)?;
    let tensor_ref = torch::eye(
        n,
        m,
        options.dtype,
        options.layout,
        options.device,
        options.requires_grad,
        options.pin_memory,
        options.memory_format,
    );
    wrap_tensor(tensor_ref, env, args)
}