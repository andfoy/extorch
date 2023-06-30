
use crate::native::torch;
use crate::shared_types::{Size, AtomString};

use rustler::{Atom, Env, Term, Encoder, Decoder, NifResult, Error};
use rustler::types::tuple::{get_tuple, make_tuple};


impl<'a> Decoder<'a> for torch::Device {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let mut device_name: String;
        let mut device_index: i64;

        match get_tuple(term) {
            Ok(device_tuple) => {
                device_name = device_tuple[0].atom_to_string()?;
                device_index = device_tuple[1].decode()?;
            },
            Err(_) => {
                let ex_device: String;
                match term.atom_to_string() {
                    Ok(dev) => {
                        ex_device = dev;
                    }
                    Err(_err) => {
                        ex_device = term.decode()?;
                    }
                }

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

        Ok(torch::Device {
            device: device_name, index: device_index
        })
    }
}

impl Encoder for torch::Device {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        let atom_device = Atom::from_str(env, &self.device).unwrap();
        let mut return_term: Term<'a> = atom_device.encode(env);
        if self.index != -1 {
            let enc_index = self.index.encode(env);
            return_term = make_tuple(env, &[return_term, enc_index]);
        }
        return_term
    }
}


impl<'a> Decoder<'a> for AtomString {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let str_name: String = term.atom_to_string()?;
        Ok(AtomString { name: str_name })
    }
}

impl Encoder for AtomString {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        Atom::from_str(env, &self.name).unwrap().encode(env)
    }
}

impl From<String> for AtomString {
    fn from(value: String) -> Self {
        AtomString { name: value }
    }
}

impl<'a> Decoder<'a> for Size {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let ex_sizes: Vec<Term<'a>>;
        ex_sizes = match get_tuple(term) {
            Ok(tup) => {
                tup
            },
            Err(_) => {
                term.decode().unwrap()
            }
        };

        let mut sizes: Vec<i64> = Vec::new();
        for term in ex_sizes {
            sizes.push(term.decode()?);
        }
        Ok(Size {
            size: sizes
        })
    }
}

impl Encoder for Size {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        let terms: Vec<Term<'a>> = self.size.iter().map(|x| x.encode(env)).collect();
        make_tuple(env, &terms[..])
    }
}


impl From<&[i64]> for Size {
    fn from(value: &[i64]) -> Self {
        let vec = value.to_vec();
        Size {
            size: vec
        }
    }
}

trait EncodeTorchScalar<'a>: Decoder<'a> {
    fn encode_scalar(term: Term<'a>) -> NifResult<torch::Scalar>;
}

impl<'a> EncodeTorchScalar<'a> for bool {
    fn encode_scalar(term: Term<'a>) -> NifResult<torch::Scalar> {
        let bool_value: NifResult<bool> = term.decode();
        match bool_value {
            Ok(value) => {
                let mut repr_vec: Vec<u8> = Vec::new();
                repr_vec.push(value.into());
                Ok(torch::Scalar {
                    _repr: repr_vec,
                    entry_used: "bool".to_owned()
                })

            },
            Err(err) => Err(err)
        }
    }
}

macro_rules! impl_scalar_for_types {
    ($(($t:ty, $dtype:tt)),*) => {
        $(
            impl<'a> EncodeTorchScalar<'a> for $t {
                fn encode_scalar(term: Term<'a>) -> NifResult<torch::Scalar> {
                    let value: NifResult<$t> = term.decode();
                    match value {
                        Ok(act_value) => {
                            let bytes = act_value.to_ne_bytes();
                            let repr: Vec<u8> = bytes.to_vec();
                            Ok(torch::Scalar {
                                _repr: repr,
                                entry_used: $dtype.to_owned()
                            })
                        },
                        Err(err) => Err(err)
                    }
                }
            }
        )*
    };
}

macro_rules! nested_type_encoding {
    ($term:ident, $t:ty $(, $rest:ty)+) => {
        match <$t>::encode_scalar($term) {
            Ok(value) => Ok(value),
            Err(_) => {
                nested_type_encoding!($term, $($rest),+)
            }
        }
    };
    ($term:ident, $t:ty) => {
        match <$t>::encode_scalar($term) {
            Ok(value) => Ok(value),
            Err(err) => {
                Err(err)
            }
        }
    }
}

impl_scalar_for_types!(
    (i8, "int8"), (i16, "int16"), (i32, "int32"), (i64, "int64"), (u8, "uint8"),
    (u16, "uint16"), (u32, "uint32"), (u64, "uint64"), (f32, "float32"),
    (f64, "float64"));


impl<'a> Decoder<'a> for torch::Scalar {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        // let mut result: NifResult<Self> = Err(Error::RaiseAtom("invalid_type"));
        let result = nested_type_encoding!(
            term, bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

        match result {
            Ok(_) => result,
            Err(_) => Err(Error::RaiseAtom("invalid_scalar_type"))
        }

    }
}
