use crate::encoding::jit::{ivalue_flat_to_term, named_tensors_to_term};
use crate::native::torch;
use crate::shared_types::{JitModuleStruct, Reference, TensorStruct};

use rustler::{Atom, Encoder, Env, Error, NifResult, ResourceArc, Term};

/// Build a TensorList from a slice of TensorStructs.
fn make_tensor_list(inputs: &[TensorStruct]) -> torch::TensorList {
    let values: Vec<torch::TensorOut> = inputs
        .iter()
        .map(|t| torch::TensorOut {
            tensor: t.resource.tensor.clone(),
            used: true,
        })
        .collect();
    torch::TensorList { values, used: true }
}

/// Helper to convert a cxx error into a NifResult error.
fn cxx_err_to_nif(err: cxx::Exception) -> Error {
    let err_msg = err.what().to_owned();
    let err_parts: Vec<&str> = err_msg.split('\n').collect();
    Error::RaiseTerm(Box::new(err_parts[0].to_owned()))
}

/// Clone a cxx Device struct (cxx shared structs don't auto-derive Clone).
fn clone_device(device: &torch::Device) -> torch::Device {
    torch::Device {
        device: device.device.clone(),
        index: device.index,
    }
}

/// Load a TorchScript model from a file path.
#[rustler::nif]
pub fn jit_load<'a>(
    path: String,
    device: torch::Device,
) -> NifResult<JitModuleStruct<'a>> {
    let elixir_device = clone_device(&device);
    let module = torch::jit_load(path, device).map_err(cxx_err_to_nif)?;
    let wrapped = torch::CrossModuleRef { module };
    let resource = ResourceArc::new(wrapped);
    Ok(JitModuleStruct {
        resource,
        reference: Reference::new(),
        device: elixir_device,
    })
}

/// Save a TorchScript model to a file path.
#[rustler::nif]
pub fn jit_save(model: JitModuleStruct, path: String) -> NifResult<()> {
    torch::jit_save(&model.resource.module, path).map_err(cxx_err_to_nif)
}

/// Run the forward method on a JIT model, returning an Elixir term.
#[rustler::nif]
pub fn jit_forward<'a>(
    env: Env<'a>,
    model: JitModuleStruct<'a>,
    inputs: Vec<TensorStruct<'a>>,
) -> NifResult<Term<'a>> {
    let input_list = make_tensor_list(&inputs);
    let flat = torch::jit_forward(&model.resource.module, input_list).map_err(cxx_err_to_nif)?;
    Ok(ivalue_flat_to_term(env, &flat))
}

/// Invoke a named method on a JIT model.
#[rustler::nif]
pub fn jit_invoke_method<'a>(
    env: Env<'a>,
    model: JitModuleStruct<'a>,
    method_name: String,
    inputs: Vec<TensorStruct<'a>>,
) -> NifResult<Term<'a>> {
    let input_list = make_tensor_list(&inputs);
    let flat = torch::jit_invoke_method(&model.resource.module, method_name, input_list)
        .map_err(cxx_err_to_nif)?;
    Ok(ivalue_flat_to_term(env, &flat))
}

/// Get names of all methods on a JIT model.
#[rustler::nif]
pub fn jit_get_method_names(model: JitModuleStruct) -> NifResult<Vec<String>> {
    let names = torch::jit_get_method_names(&model.resource.module).map_err(cxx_err_to_nif)?;
    Ok(names.into_iter().map(|s| s.to_string()).collect())
}

/// Get named parameters of a JIT model as a list of {name, tensor} tuples.
#[rustler::nif]
pub fn jit_named_parameters<'a>(env: Env<'a>, model: JitModuleStruct<'a>) -> NifResult<Term<'a>> {
    let params =
        torch::jit_named_parameters(&model.resource.module).map_err(cxx_err_to_nif)?;
    Ok(named_tensors_to_term(env, &params))
}

/// Get named buffers of a JIT model as a list of {name, tensor} tuples.
#[rustler::nif]
pub fn jit_named_buffers<'a>(env: Env<'a>, model: JitModuleStruct<'a>) -> NifResult<Term<'a>> {
    let buffers =
        torch::jit_named_buffers(&model.resource.module).map_err(cxx_err_to_nif)?;
    Ok(named_tensors_to_term(env, &buffers))
}

/// Get names of submodules of a JIT model.
#[rustler::nif]
pub fn jit_named_modules(model: JitModuleStruct) -> NifResult<Vec<String>> {
    let names = torch::jit_named_modules(&model.resource.module).map_err(cxx_err_to_nif)?;
    Ok(names.into_iter().map(|s| s.to_string()).collect())
}

/// Set a JIT model to evaluation mode.
#[rustler::nif]
pub fn jit_set_eval(model: JitModuleStruct) -> NifResult<()> {
    torch::jit_set_eval(&model.resource.module).map_err(cxx_err_to_nif)
}

/// Set a JIT model to training mode.
#[rustler::nif]
pub fn jit_set_train(model: JitModuleStruct) -> NifResult<()> {
    torch::jit_set_train(&model.resource.module).map_err(cxx_err_to_nif)
}

/// Move a JIT model to a different device.
#[rustler::nif]
pub fn jit_to_device<'a>(
    model: JitModuleStruct<'a>,
    device: torch::Device,
) -> NifResult<JitModuleStruct<'a>> {
    let elixir_device = clone_device(&device);
    let module =
        torch::jit_to_device(&model.resource.module, device).map_err(cxx_err_to_nif)?;
    let wrapped = torch::CrossModuleRef { module };
    let resource = ResourceArc::new(wrapped);
    Ok(JitModuleStruct {
        resource,
        reference: Reference::new(),
        device: elixir_device,
    })
}

// ============================================================================
// IR Introspection
// ============================================================================

/// Get the JIT graph IR as a string.
#[rustler::nif]
pub fn jit_graph_str(model: JitModuleStruct) -> NifResult<String> {
    let graph = torch::jit_graph_str(&model.resource.module).map_err(cxx_err_to_nif)?;
    Ok(graph.to_string())
}

/// Get parameter info for a JIT module.
/// Returns a list of maps: %{name, shape, dtype, requires_grad}
#[rustler::nif]
pub fn jit_module_parameters_info<'a>(env: Env<'a>, model: JitModuleStruct<'a>) -> NifResult<Term<'a>> {
    let params = torch::jit_module_parameters_info(&model.resource.module).map_err(cxx_err_to_nif)?;

    let result: Vec<Term<'a>> = params.iter().map(|p| {
        let name: &str = p.name.as_str();
        let shape: Vec<i64> = p.shape.iter().copied().collect();
        let dtype_str: &str = p.dtype.as_str();
        let dtype_atom = Atom::from_str(env, dtype_str).unwrap();

        let keys = vec![
            Atom::from_str(env, "name").unwrap().encode(env),
            Atom::from_str(env, "shape").unwrap().encode(env),
            Atom::from_str(env, "dtype").unwrap().encode(env),
            Atom::from_str(env, "requires_grad").unwrap().encode(env),
        ];
        let values = vec![
            name.encode(env),
            shape.encode(env),
            dtype_atom.encode(env),
            p.requires_grad.encode(env),
        ];

        Term::map_from_arrays(env, &keys, &values).unwrap()
    }).collect();

    Ok(result.encode(env))
}

/// Get submodule info for a JIT module.
/// Returns a list of maps: %{name, type_name, parameters}
#[rustler::nif]
pub fn jit_module_submodules_info<'a>(env: Env<'a>, model: JitModuleStruct<'a>) -> NifResult<Term<'a>> {
    let submodules = torch::jit_module_submodules_info(&model.resource.module).map_err(cxx_err_to_nif)?;

    let result: Vec<Term<'a>> = submodules.iter().map(|s| {
        let name: &str = s.name.as_str();
        let type_name: &str = s.type_name.as_str();

        let params: Vec<Term<'a>> = s.parameters.iter().map(|p| {
            let pname: &str = p.name.as_str();
            let shape: Vec<i64> = p.shape.iter().copied().collect();
            let dtype_str: &str = p.dtype.as_str();
            let dtype_atom = Atom::from_str(env, dtype_str).unwrap();

            let keys = vec![
                Atom::from_str(env, "name").unwrap().encode(env),
                Atom::from_str(env, "shape").unwrap().encode(env),
                Atom::from_str(env, "dtype").unwrap().encode(env),
                Atom::from_str(env, "requires_grad").unwrap().encode(env),
            ];
            let values = vec![
                pname.encode(env),
                shape.encode(env),
                dtype_atom.encode(env),
                p.requires_grad.encode(env),
            ];

            Term::map_from_arrays(env, &keys, &values).unwrap()
        }).collect();

        let keys = vec![
            Atom::from_str(env, "name").unwrap().encode(env),
            Atom::from_str(env, "type_name").unwrap().encode(env),
            Atom::from_str(env, "parameters").unwrap().encode(env),
        ];
        let values = vec![
            name.encode(env),
            type_name.encode(env),
            params.encode(env),
        ];

        Term::map_from_arrays(env, &keys, &values).unwrap()
    }).collect();

    Ok(result.encode(env))
}

/// Get method names for a JIT module.
#[rustler::nif]
pub fn jit_module_methods_info(model: JitModuleStruct) -> NifResult<Vec<String>> {
    let methods = torch::jit_module_methods_info(&model.resource.module).map_err(cxx_err_to_nif)?;
    Ok(methods.into_iter().map(|s| s.to_string()).collect())
}
