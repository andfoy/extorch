
extern crate rustler;
use rustler::{NifResult, Term, Error, Env};

use crate::native::{torch};
use crate::rustler::Encoder;
use crate::conversion::dtypes::{ALL_TYPES};


fn find_scalar<'a>(value: Term<'a>) -> Result<torch::Scalar, Error> {
    let bool_value: NifResult<bool> = value.decode();
    match bool_value {
        Ok(value) => {
            let scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: value,
                entry_used: "bool".to_owned(),
            };
            Ok(scalar_result)
        }
        Err(_err) => find_ui8(value),
    }
}

fn find_ui8<'a>(value: Term<'a>) -> Result<torch::Scalar, Error> {
    let u8_value: NifResult<u8> = value.decode();
    match u8_value {
        Ok(value) => {
            let scalar_result = torch::Scalar {
                _ui8: value,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: "uint8".to_owned(),
            };
            Ok(scalar_result)
        }
        Err(_err) => find_i32(value),
    }
}

fn find_i32<'a>(value: Term<'a>) -> Result<torch::Scalar, Error> {
    let i32_value: NifResult<i32> = value.decode();
    match i32_value {
        Ok(value) => {
            let scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: value,
                _i64: -1,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: "int32".to_owned(),
            };
            Ok(scalar_result)
        }
        Err(_err) => find_i64(value),
    }
}

fn find_i64<'a>(value: Term<'a>) -> Result<torch::Scalar, Error> {
    let i64_value: NifResult<i64> = value.decode();
    match i64_value {
        Ok(value) => {
            let scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: value,
                _f16: -1.0,
                _f32: -1.0,
                _f64: -1.0,
                _bool: false,
                entry_used: "int64".to_owned(),
            };
            Ok(scalar_result)
        }
        Err(_err) => find_f32(value),
    }
}

fn find_f32<'a>(value: Term<'a>) -> Result<torch::Scalar, Error> {
    let f32_value: NifResult<f32> = value.decode();
    match f32_value {
        Ok(value) => {
            let scalar_result = torch::Scalar {
                _ui8: 0,
                _i8: -1,
                _i16: -1,
                _i32: -1,
                _i64: -1,
                _f16: -1.0,
                _f32: value,
                _f64: -1.0,
                _bool: false,
                entry_used: "float32".to_owned(),
            };
            Ok(scalar_result)
        }
        Err(_err) => find_f64(value),
    }
}

fn find_f64<'a>(value: Term<'a>) -> Result<torch::Scalar, Error> {
    let f64_value: f64 = value.decode()?;
    let scalar_result = torch::Scalar {
        _ui8: 0,
        _i8: -1,
        _i16: -1,
        _i32: -1,
        _i64: -1,
        _f16: -1.0,
        _f32: -1.0,
        _f64: f64_value,
        _bool: false,
        entry_used: "float64".to_owned(),
    };
    Ok(scalar_result)
}

pub fn unpack_scalar<'a>(
    index: usize,
    _env: Env<'a>,
    args: &[Term<'a>],
) -> Result<torch::Scalar, Error> {
    let ex_scalar: Term<'a> = args[index];
    find_scalar(ex_scalar)
}

pub fn unpack_scalar_typed<'a>(
    ex_scalar: Term<'a>,
    s_type: &str
) -> Result<torch::Scalar, Error> {
    let mut type_cast: String = "float32".to_owned();
    if ALL_TYPES.contains_key::<str>(s_type) {
        type_cast = ALL_TYPES.get::<str>(s_type).unwrap().to_string();
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

pub fn wrap_scalar_typed<'a>(
    scalar: torch::Scalar,
    env: Env<'a>,
) -> Result<Term<'a>, Error> {
    // let mut value: f32 = 0.0;
    let result: Term<'a>;
    match &scalar.entry_used[..] {
        "uint8" => {
            let value = scalar._ui8;
            result = value.encode(env);
        }
        "int8" => {
            let value = scalar._i8;
            result = value.encode(env);
        }
        "int16" => {
            let value = scalar._i16;
            result = value.encode(env);
        }
        "int32" => {
            let value = scalar._i32;
            result = value.encode(env);
        }
        "int64" => {
            let value = scalar._i64;
            result = value.encode(env);
        }
        "float16" => {
            let value = scalar._f16;
            result = value.encode(env);
        }
        "bfloat16" => {
            let value = scalar._f16;
            result = value.encode(env);
        }
        "float32" => {
            let value = scalar._f32;
            result = value.encode(env);
        }
        "float64" => {
            let value = scalar._f64;
            result = value.encode(env);
        }
        "bool" => {
            let value = scalar._bool;
            result = value.encode(env);
        }
        _ => {
            let value = scalar._f32;
            result = value.encode(env);
        }
    }
    Ok(result)
}
