use crate::native::torch;
use crate::shared_types::{
    AtomString, Complex, ExPrintOptions, ExSlice, ListWrapper, Size, TensorIndex, TensorStruct,
};

use rustler::types::tuple::{get_tuple, make_tuple};
use rustler::{Atom, Decoder, Encoder, Env, Error, NifResult, Term};

use cxx::SharedPtr;

impl<'a> Decoder<'a> for torch::Device {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let mut device_name: String;
        let mut device_index: i64;

        match get_tuple(term) {
            Ok(device_tuple) => {
                device_name = device_tuple[0].atom_to_string()?;
                device_index = device_tuple[1].decode()?;
            }
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
                    match device_parts[1].parse() {
                        Ok(idx) => {
                            device_index = idx;
                        },
                        Err(_) => {
                            return Err(Error::RaiseAtom("Bad device index"));
                        }
                    }
                }
            }
        }

        Ok(torch::Device {
            device: device_name,
            index: device_index,
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
        let partial_ex_sizes = match get_tuple(term) {
            Ok(tup) => Ok(tup),
            Err(_) => term.decode(),
        };

        match partial_ex_sizes {
            Ok(ex_sizes) => {
                let mut sizes: Vec<i64> = Vec::new();
                for term in ex_sizes {
                    sizes.push(term.decode()?);
                }
                Ok(Size { size: sizes })
            },
            Err(err) => Err(err)
        }

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
        Size { size: vec }
    }
}

trait EncodeTorchScalar<'a>: Decoder<'a> {
    fn encode_scalar(term: Term<'a>) -> NifResult<torch::Scalar>;
}

trait DecodeTorchScalar: Encoder {
    fn decode_scalar<'a>(scalar: &torch::Scalar, env: Env<'a>) -> Term<'a>;
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
                    entry_used: "bool".to_owned(),
                })
            }
            Err(err) => Err(err),
        }
    }
}

fn match_special_atoms<'a>(term: Term<'a>) -> Result<f64, Error> {
    match term.atom_to_string()?.as_str() {
        "nan" => Ok(f64::NAN),
        "inf" => Ok(f64::INFINITY),
        "ninf" => Ok(f64::NEG_INFINITY),
        _ => Err(Error::RaiseAtom("invalid_atom_value"))
    }
}

impl<'a> EncodeTorchScalar<'a> for Complex<'a> {
    fn encode_scalar(term: Term<'a>) -> NifResult<torch::Scalar> {
        let complex_value: NifResult<Complex> = term.decode();
        match complex_value {
            Ok(value) => {
                let real_value = match value.real.is_atom() {
                    true => match_special_atoms(value.real)?,
                    false => value.real.decode::<f64>()?
                };

                let im_value = match value.imaginary.is_atom() {
                    true => match_special_atoms(value.imaginary)?,
                    false => value.imaginary.decode::<f64>()?
                };

                let re_bytes = real_value.to_ne_bytes();
                let im_bytes = im_value.to_ne_bytes();
                let mut repr: Vec<u8> = re_bytes.to_vec();
                repr.append(&mut im_bytes.to_vec());
                Ok(torch::Scalar {
                    _repr: repr,
                    entry_used: "complex128".to_string(),
                })
            }
            Err(err) => Err(err),
        }
    }
}

impl<'a> EncodeTorchScalar<'a> for AtomString {
    fn encode_scalar(term: Term<'a>) -> NifResult<torch::Scalar> {
        let atom_value: NifResult<AtomString> = term.decode();
        match atom_value {
            Ok(atom_str) => {
                let value = match atom_str.name.as_str() {
                    "nan" => Some(f32::NAN),
                    "inf" => Some(f32::INFINITY),
                    "ninf" => Some(f32::NEG_INFINITY),
                    _ => None
                };

                match value {
                    Some(val) => {
                        let bytes = val.to_ne_bytes();
                        let repr: Vec<u8> = bytes.to_vec();
                        Ok(torch::Scalar {
                            _repr: repr,
                            entry_used: "float32".to_owned()
                        })
                    },
                    None => Err(Error::RaiseAtom("invalid numeric atom"))
                }
            },
            Err(err) => Err(err)
        }
    }
}


impl DecodeTorchScalar for bool {
    fn decode_scalar<'a>(scalar: &torch::Scalar, env: Env<'a>) -> Term<'a> {
        let value = scalar._repr.get(0).unwrap();
        let bool_value = *value != 0u8;
        bool_value.encode(env)
    }
}

fn pack_possible_special_atom<'a>(value: f64, env: Env<'a>) -> Term<'a> {
    let inf_value = match value.is_infinite() && value.is_sign_positive() {
        true => Some(Atom::from_str(env, "inf").unwrap().to_term(env)),
        false => None
    };

    let ninf_value= match value.is_infinite() && value.is_sign_negative() {
        true => Some(Atom::from_str(env, "ninf").unwrap().to_term(env)),
        false => inf_value
    };

    let nan_value = match value.is_nan() {
        true => Some(Atom::from_str(env, "nan").unwrap().to_term(env)),
        false => ninf_value
    };

    match nan_value {
        Some(value) => value,
        None => value.encode(env)
    }
}

impl DecodeTorchScalar for Complex<'_> {
    fn decode_scalar<'a>(scalar: &torch::Scalar, env: Env<'a>) -> Term<'a> {
        let real_part = &scalar._repr[..8];
        let im_part = &scalar._repr[8..];
        let real = f64::from_ne_bytes(real_part.try_into().unwrap());
        let im = f64::from_ne_bytes(im_part.try_into().unwrap());

        let complex = Complex {
            real: pack_possible_special_atom(real, env),
            imaginary: pack_possible_special_atom(im, env),
        };
        complex.encode(env)
    }
}

macro_rules! impl_full_scalar_for_types {
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

            impl DecodeTorchScalar for $t {
                fn decode_scalar<'a>(scalar: &torch::Scalar, env: Env<'a>) -> Term<'a> {
                    let slice = &scalar._repr[..];
                    let value = <$t>::from_ne_bytes(slice.try_into().unwrap());
                    value.encode(env)
                }
            }
        )*
    };
}

macro_rules! impl_float_scalar_for_types {
    ($(($t:ty, $dtype:tt)),*) => {
        $(
            impl<'a> EncodeTorchScalar<'a> for $t {
                fn encode_scalar(term: Term<'a>) -> NifResult<torch::Scalar> {
                    let atom_value: NifResult<AtomString> = term.decode();
                    match atom_value {
                        Ok(atom_str) => {
                            let value = match atom_str.name.as_str() {
                                "nan" => Some(<$t>::NAN),
                                "inf" => Some(<$t>::INFINITY),
                                "ninf" => Some(<$t>::NEG_INFINITY),
                                _ => None
                            };

                            match value {
                                Some(val) => {
                                    let bytes = val.to_ne_bytes();
                                    let repr: Vec<u8> = bytes.to_vec();
                                    Ok(torch::Scalar {
                                        _repr: repr,
                                        entry_used: $dtype.to_owned()
                                    })
                                },
                                None => Err(Error::RaiseAtom("invalid numeric atom"))
                            }
                        },
                        Err(_) => {
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
                }
            }

            impl DecodeTorchScalar for $t {
                fn decode_scalar<'a>(scalar: &torch::Scalar, env: Env<'a>) -> Term<'a> {
                    let slice = &scalar._repr[..];
                    let value = <$t>::from_ne_bytes(slice.try_into().unwrap());

                    let nan_value = match value.is_nan() {
                        true => Some(Atom::from_str(env, "nan").unwrap().encode(env)),
                        false => None
                    };

                    let pinf_value = match value.is_infinite() && value.is_sign_positive() {
                        true => Some(Atom::from_str(env, "inf").unwrap().encode(env)),
                        false => nan_value
                    };

                    let ninf_value = match value.is_infinite() && value.is_sign_negative() {
                        true => Some(Atom::from_str(env, "ninf").unwrap().encode(env)),
                        false => pinf_value
                    };

                    match ninf_value {
                        Some(term) => term,
                        None => value.encode(env)
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

impl_full_scalar_for_types!(
    (i8, "int8"),
    (i16, "int16"),
    (i32, "int32"),
    (i64, "int64"),
    (u8, "uint8"),
    (u16, "uint16"),
    (u32, "uint32"),
    (u64, "uint64")
);

impl_float_scalar_for_types!(
    (f32, "float32"),
    (f64, "float64")
);

impl<'a> Decoder<'a> for torch::Scalar {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        // let mut result: NifResult<Self> = Err(Error::RaiseAtom("invalid_type"));
        let result = nested_type_encoding!(
            term, bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, Complex, AtomString
        );

        match result {
            Ok(_) => result,
            Err(_) => Err(Error::RaiseAtom("invalid_scalar_type")),
        }
    }
}

impl Encoder for torch::Scalar {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        match self.entry_used.as_str() {
            "bool" => bool::decode_scalar(self, env),
            "int8" => i8::decode_scalar(self, env),
            "int16" => i16::decode_scalar(self, env),
            "int32" => i32::decode_scalar(self, env),
            "int64" => i64::decode_scalar(self, env),
            "uint8" => u8::decode_scalar(self, env),
            "uint16" => u16::decode_scalar(self, env),
            "uint32" => u32::decode_scalar(self, env),
            "uint64" => u64::decode_scalar(self, env),
            "float32" => f32::decode_scalar(self, env),
            "float64" => f64::decode_scalar(self, env),
            "complex32" => Complex::decode_scalar(self, env),
            "complex64" => Complex::decode_scalar(self, env),
            "complex128" => Complex::decode_scalar(self, env),
            &_ => Atom::from_str(env, "not_converted").unwrap().encode(env),
        }
    }
}

fn decode_coerced_scalar<'a>(term: Term<'a>, coerce_dtype: &String) -> NifResult<torch::Scalar> {
    match coerce_dtype.as_str() {
        "bool" => bool::encode_scalar(term),
        "int8" => i8::encode_scalar(term),
        "int16" => i16::encode_scalar(term),
        "int32" => i32::encode_scalar(term),
        "int64" => i64::encode_scalar(term),
        "uint8" => u8::encode_scalar(term),
        "uint16" => u16::encode_scalar(term),
        "uint32" => u32::encode_scalar(term),
        "uint64" => u64::encode_scalar(term),
        "float32" => f32::encode_scalar(term),
        "float64" => f64::encode_scalar(term),
        "complex32" => Complex::encode_scalar(term),
        "complex64" => Complex::encode_scalar(term),
        "complex128" => Complex::encode_scalar(term),
        &_ => Err(Error::RaiseAtom("not_converted")),
    }
}

impl<'a> Decoder<'a> for torch::ScalarList {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let list_wrapper: ListWrapper<'a> = term.decode()?;
        let dtype: String = list_wrapper.dtype.atom_to_string()?;
        let scalar_vec: Vec<torch::Scalar> = list_wrapper
            .list
            .iter()
            .map(|x| decode_coerced_scalar(*x, &dtype).unwrap())
            .collect();
        let size: Size = list_wrapper.size.decode()?;
        Ok(Self {
            list: scalar_vec,
            size: size.size,
        })
    }
}

impl Encoder for torch::ScalarList {
    fn encode<'a>(&self, env: Env<'a>) -> Term<'a> {
        let mut dim_count: i64 = 0;
        let mut cur_parent: Vec<Term<'a>> = Vec::new();

        let mut in_stack: Vec<Term<'a>> = self.list.iter().map(|x| x.encode(env)).collect();
        let mut out_stack: Vec<Term<'a>> = Vec::new();

        for dim_size in self.size.iter().rev() {
            for term in &in_stack[..] {
                cur_parent.push(*term);
                dim_count = dim_count + 1;
                cur_parent = match dim_count == *dim_size {
                    true => {
                        dim_count = 0;
                        out_stack.push(cur_parent.encode(env));
                        Vec::<Term<'a>>::new()
                    }
                    false => cur_parent,
                }
            }
            in_stack = out_stack;
            out_stack = Vec::new();
        }

        match in_stack.get(0) {
            Some(val) => val.encode(env),
            None => in_stack.encode(env),
        }
    }
}

impl<'a> Decoder<'a> for torch::TorchSlice {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let wrapped_term: ExSlice = term.decode()?;
        Ok(torch::TorchSlice {
            start: wrapped_term.start,
            stop: wrapped_term.stop,
            step: wrapped_term.step,
            enc: wrapped_term.mask,
        })
    }
}

impl<'a> Decoder<'a> for torch::TorchIndex {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let atom_value = match term.atom_to_string() {
            Ok(value) => match value.as_str() {
                "nil" => Some(Ok(torch::TorchIndex {
                    integer: -1,
                    slice: torch::TorchSlice {
                        start: 0,
                        stop: 0,
                        step: 1,
                        enc: 0,
                    },
                    boolean: false,
                    tensor: SharedPtr::<torch::CrossTensor>::null(),
                    type_: 0,
                })),
                "ellipsis" => Some(Ok(torch::TorchIndex {
                    integer: -1,
                    slice: torch::TorchSlice {
                        start: 0,
                        stop: 0,
                        step: 1,
                        enc: 0,
                    },
                    boolean: false,
                    tensor: SharedPtr::<torch::CrossTensor>::null(),
                    type_: 1,
                })),
                "true" => Some(Ok(torch::TorchIndex {
                    integer: -1,
                    slice: torch::TorchSlice {
                        start: 0,
                        stop: 0,
                        step: 1,
                        enc: 0,
                    },
                    boolean: true,
                    tensor: SharedPtr::<torch::CrossTensor>::null(),
                    type_: 3,
                })),
                "false" => Some(Ok(torch::TorchIndex {
                    integer: -1,
                    slice: torch::TorchSlice {
                        start: 0,
                        stop: 0,
                        step: 1,
                        enc: 0,
                    },
                    boolean: false,
                    tensor: SharedPtr::<torch::CrossTensor>::null(),
                    type_: 3,
                })),
                _ => Some(Err(Error::RaiseAtom(
                    "only ellipsis and nil are valid indices",
                ))),
            },
            Err(_) => None,
        };

        let int_value = match atom_value {
            Some(_) => atom_value,
            None => {
                let index_val: Result<i64, Error> = term.decode();
                match index_val {
                    Ok(int_value) => Some(Ok(torch::TorchIndex {
                        integer: int_value,
                        slice: torch::TorchSlice {
                            start: 0,
                            stop: 0,
                            step: 1,
                            enc: 0,
                        },
                        boolean: false,
                        tensor: SharedPtr::<torch::CrossTensor>::null(),
                        type_: 2,
                    })),
                    Err(_) => None,
                }
            }
        };

        let slice_value = match int_value {
            Some(_) => int_value,
            None => {
                let slice_val: Result<torch::TorchSlice, Error> = term.decode();
                match slice_val {
                    Ok(slice_value) => Some(Ok(torch::TorchIndex {
                        integer: -1,
                        slice: slice_value,
                        boolean: false,
                        tensor: SharedPtr::<torch::CrossTensor>::null(),
                        type_: 4,
                    })),
                    Err(_) => None,
                }
            }
        };

        match slice_value {
            Some(value) => value,
            None => {
                let tensor_val: Result<TensorStruct<'a>, Error> = term.decode();
                match tensor_val {
                    Ok(tensor) => Ok(torch::TorchIndex {
                        integer: -1,
                        slice: torch::TorchSlice {
                            start: 0,
                            stop: 0,
                            step: 1,
                            enc: 0,
                        },
                        boolean: false,
                        tensor: tensor.resource.tensor.clone(),
                        type_: 5,
                    }),
                    Err(e) => Err(e),
                }
            }
        }
    }
}

impl<'a> Decoder<'a> for TensorIndex {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let indices: Vec<torch::TorchIndex> = term.decode()?;
        Ok(TensorIndex { indices })
    }
}

impl<'a> Decoder<'a> for torch::PrintOptions {
    fn decode(term: Term<'a>) -> NifResult<Self> {
        let print_opts: ExPrintOptions = term.decode()?;
        let sci_mode = match print_opts.sci_mode {
            Some(value) => (value as u8) + 1,
            None => 0,
        };
        Ok(torch::PrintOptions {
            precision: print_opts.precision,
            threshold: print_opts.threshold,
            edgeitems: print_opts.edgeitems,
            linewidth: print_opts.linewidth,
            sci_mode,
        })
    }
}
