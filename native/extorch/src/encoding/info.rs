
use std::str::FromStr;

use crate::native::torch;
use crate::shared_types::{Size, AtomString};

use rustler::{Atom, Env, Term, Encoder, Decoder, NifResult};
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
