use crate::native::torch;
use crate::shared_types::{AOTIModelStruct, Reference, TensorStruct};

use rustler::{Encoder, Env, Error, NifResult, ResourceArc, Term};

fn cxx_err(err: cxx::Exception) -> Error {
    let msg = err.what().to_owned();
    let parts: Vec<&str> = msg.split('\n').collect();
    Error::RaiseTerm(Box::new(parts[0].to_owned()))
}

#[rustler::nif]
pub fn aoti_is_available() -> NifResult<bool> {
    torch::aoti_is_available().map_err(cxx_err)
}

#[rustler::nif]
pub fn aoti_load<'a>(path: String, model_name: String, device_index: i64) -> NifResult<AOTIModelStruct<'a>> {
    let loader = torch::aoti_load(path, model_name, device_index).map_err(cxx_err)?;
    let wrapped = torch::CrossAOTILoaderRef { loader };
    let resource = ResourceArc::new(wrapped);
    Ok(AOTIModelStruct {
        resource,
        reference: Reference::new(),
    })
}

#[rustler::nif]
pub fn aoti_forward<'a>(env: Env<'a>, model: AOTIModelStruct<'a>, inputs: Vec<TensorStruct<'a>>) -> NifResult<Term<'a>> {
    let values: Vec<torch::TensorOut> = inputs
        .iter()
        .map(|t| torch::TensorOut {
            tensor: t.resource.tensor.clone(),
            used: true,
        })
        .collect();
    let input_list = torch::TensorList { values, used: true };

    let result = torch::aoti_forward(&model.resource.loader, input_list).map_err(cxx_err)?;

    // TensorList → list of TensorStruct
    let out: Vec<Term<'a>> = result
        .values
        .iter()
        .filter(|t| t.used)
        .map(|t| {
            let ts: TensorStruct<'a> = t.tensor.clone().into();
            ts.encode(env)
        })
        .collect();

    Ok(out.encode(env))
}

#[rustler::nif]
pub fn aoti_get_metadata_keys(model: AOTIModelStruct) -> NifResult<Vec<String>> {
    let keys = torch::aoti_get_metadata_keys(&model.resource.loader).map_err(cxx_err)?;
    Ok(keys.into_iter().map(|s| s.to_string()).collect())
}

#[rustler::nif]
pub fn aoti_get_metadata_value(model: AOTIModelStruct, key: String) -> NifResult<String> {
    let val = torch::aoti_get_metadata_value(&model.resource.loader, key).map_err(cxx_err)?;
    Ok(val.to_string())
}

#[rustler::nif]
pub fn aoti_get_constant_fqns(model: AOTIModelStruct) -> NifResult<Vec<String>> {
    let fqns = torch::aoti_get_constant_fqns(&model.resource.loader).map_err(cxx_err)?;
    Ok(fqns.into_iter().map(|s| s.to_string()).collect())
}
