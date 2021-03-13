extern crate rustler;
use rustler::types::tuple::{get_tuple, make_tuple};
use rustler::types::Atom;
use rustler::{Env, Error, Term};

use crate::native::torch;
use crate::rustler::Encoder;
use crate::shared_types::TensorOptions;

pub fn unpack_size_init<'a>(
    index: usize,
    _env: Env<'a>,
    args: &[Term<'a>],
) -> Result<Vec<i64>, Error> {
    let tuple_sizes: Term<'a> = args[index];
    unpack_size(tuple_sizes)
}

pub fn unpack_size<'a>(tuple_sizes: Term<'a>) -> Result<Vec<i64>, Error> {
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

pub fn unpack_tensor_options<'a>(
    off: usize,
    _env: Env<'a>,
    args: &[Term<'a>],
) -> Result<TensorOptions, Error> {
    let ex_dtype: String = args[off].atom_to_string()?;
    let ex_layout: String = args[off + 1].atom_to_string()?;
    // let ex_device: String = args[3].atom_to_string()?;
    let tuple_device: Term<'a> = args[off + 2];

    let mut device_name;
    let mut device_index: i64;

    match get_tuple(tuple_device) {
        Ok(device_tuple) => {
            device_name = device_tuple[0].atom_to_string()?;
            device_index = device_tuple[1].decode()?;
        }

        Err(_err) => {
            let ex_device: String;
            match args[off + 2].atom_to_string() {
                Ok(dev) => {
                    ex_device = dev;
                }
                Err(_err) => {
                    ex_device = args[off + 2].decode()?;
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

    let requires_grad: bool = args[off + 3].decode()?;
    let pin_memory: bool = args[off + 4].decode()?;
    let memory_format: String = args[off + 5].atom_to_string()?;

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

pub fn wrap_size<'a>(env: Env<'a>, sizes: &'static [i64]) -> Result<Term<'a>, Error> {
    let enc_sizes: Vec<Term<'a>> = sizes.iter().map(|size| size.encode(env)).collect();
    let tuple_sizes = make_tuple(env, &enc_sizes);
    Ok(tuple_sizes.encode(env))
}

pub fn wrap_str_atom<'a>(env: Env<'a>, input: String) -> Result<Term<'a>, Error> {
    let atom_str = Atom::from_str(env, &input)?;
    Ok(atom_str.encode(env))
}

pub fn wrap_device<'a>(env: Env<'a>, device: torch::Device) -> Result<Term<'a>, Error> {
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
