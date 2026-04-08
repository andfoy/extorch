use crate::encoding::jit::named_tensors_to_term;
use crate::native::torch;
use crate::shared_types::{NNModuleStruct, Reference, TensorStruct};

use rustler::{Env, Error, NifResult, ResourceArc, Term};

fn cxx_err(err: cxx::Exception) -> Error {
    let msg = err.what().to_owned();
    let parts: Vec<&str> = msg.split('\n').collect();
    Error::RaiseTerm(Box::new(parts[0].to_owned()))
}

fn wrap_nn_module<'a>(module: cxx::SharedPtr<torch::CrossNNModule>) -> NifResult<NNModuleStruct<'a>> {
    let type_name = torch::nn_type_name(&module).map_err(cxx_err)?.to_string();
    let wrapped = torch::CrossNNModuleRef { module };
    let resource = ResourceArc::new(wrapped);
    Ok(NNModuleStruct {
        resource,
        reference: Reference::new(),
        type_name,
    })
}

// Layer factories

#[rustler::nif]
pub fn nn_linear<'a>(in_features: i64, out_features: i64, bias: bool) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_linear(in_features, out_features, bias).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_conv1d<'a>(
    in_channels: i64, out_channels: i64, kernel_size: i64,
    stride: i64, padding: i64, dilation: i64, groups: i64, bias: bool,
) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_conv2d<'a>(
    in_channels: i64, out_channels: i64, kernel_size: i64,
    stride: i64, padding: i64, dilation: i64, groups: i64, bias: bool,
) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_batch_norm1d<'a>(
    num_features: i64, eps: f64, momentum: f64, affine: bool, track_running_stats: bool,
) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_batch_norm1d(num_features, eps, momentum, affine, track_running_stats).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_batch_norm2d<'a>(
    num_features: i64, eps: f64, momentum: f64, affine: bool, track_running_stats: bool,
) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_batch_norm2d(num_features, eps, momentum, affine, track_running_stats).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_layer_norm<'a>(normalized_shape: Vec<i64>, eps: f64, elementwise_affine: bool) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_layer_norm(normalized_shape, eps, elementwise_affine).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_dropout<'a>(p: f64, inplace: bool) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_dropout(p, inplace).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_embedding<'a>(
    num_embeddings: i64, embedding_dim: i64, padding_idx: i64, has_padding_idx: bool,
) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_embedding(num_embeddings, embedding_dim, padding_idx, has_padding_idx).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_relu<'a>(inplace: bool) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_relu(inplace).map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_gelu<'a>() -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_gelu().map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_sigmoid<'a>() -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_sigmoid().map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_tanh<'a>() -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_tanh().map_err(cxx_err)?;
    wrap_nn_module(module)
}

#[rustler::nif]
pub fn nn_softmax<'a>(dim: i64) -> NifResult<NNModuleStruct<'a>> {
    let module = torch::nn_softmax(dim).map_err(cxx_err)?;
    wrap_nn_module(module)
}

// Additional convolutions

#[rustler::nif]
pub fn nn_conv3d<'a>(in_ch: i64, out_ch: i64, kernel: i64, stride: i64, pad: i64, dil: i64, groups: i64, bias: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_conv3d(in_ch, out_ch, kernel, stride, pad, dil, groups, bias).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_conv_transpose1d<'a>(in_ch: i64, out_ch: i64, kernel: i64, stride: i64, pad: i64, out_pad: i64, groups: i64, bias: bool, dil: i64) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_conv_transpose1d(in_ch, out_ch, kernel, stride, pad, out_pad, groups, bias, dil).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_conv_transpose2d<'a>(in_ch: i64, out_ch: i64, kernel: i64, stride: i64, pad: i64, out_pad: i64, groups: i64, bias: bool, dil: i64) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_conv_transpose2d(in_ch, out_ch, kernel, stride, pad, out_pad, groups, bias, dil).map_err(cxx_err)?)
}

// Pooling

#[rustler::nif]
pub fn nn_max_pool1d<'a>(kernel: i64, stride: i64, pad: i64, dil: i64, ceil: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_max_pool1d(kernel, stride, pad, dil, ceil).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_max_pool2d<'a>(kernel: i64, stride: i64, pad: i64, dil: i64, ceil: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_max_pool2d(kernel, stride, pad, dil, ceil).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_avg_pool1d<'a>(kernel: i64, stride: i64, pad: i64, ceil: bool, count_pad: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_avg_pool1d(kernel, stride, pad, ceil, count_pad).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_avg_pool2d<'a>(kernel: i64, stride: i64, pad: i64, ceil: bool, count_pad: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_avg_pool2d(kernel, stride, pad, ceil, count_pad).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_adaptive_avg_pool1d<'a>(output_size: i64) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_adaptive_avg_pool1d(output_size).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_adaptive_avg_pool2d<'a>(output_h: i64, output_w: i64) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_adaptive_avg_pool2d(output_h, output_w).map_err(cxx_err)?)
}

// Additional normalization

#[rustler::nif]
pub fn nn_group_norm<'a>(num_groups: i64, num_channels: i64, eps: f64, affine: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_group_norm(num_groups, num_channels, eps, affine).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_instance_norm1d<'a>(num_features: i64, eps: f64, momentum: f64, affine: bool, track: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_instance_norm1d(num_features, eps, momentum, affine, track).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_instance_norm2d<'a>(num_features: i64, eps: f64, momentum: f64, affine: bool, track: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_instance_norm2d(num_features, eps, momentum, affine, track).map_err(cxx_err)?)
}

// Recurrent

#[rustler::nif]
pub fn nn_lstm<'a>(input_size: i64, hidden_size: i64, num_layers: i64, bias: bool, batch_first: bool, dropout: f64, bidirectional: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_lstm(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_gru<'a>(input_size: i64, hidden_size: i64, num_layers: i64, bias: bool, batch_first: bool, dropout: f64, bidirectional: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_gru(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional).map_err(cxx_err)?)
}

// Transformer

#[rustler::nif]
pub fn nn_multihead_attention<'a>(embed_dim: i64, num_heads: i64, dropout: f64, bias: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_multihead_attention(embed_dim, num_heads, dropout, bias).map_err(cxx_err)?)
}

// Utility

#[rustler::nif]
pub fn nn_flatten<'a>(start_dim: i64, end_dim: i64) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_flatten(start_dim, end_dim).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_unflatten<'a>(dim: i64, sizes: Vec<i64>) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_unflatten(dim, sizes).map_err(cxx_err)?)
}

// Additional activations

#[rustler::nif]
pub fn nn_leaky_relu<'a>(negative_slope: f64, inplace: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_leaky_relu(negative_slope, inplace).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_elu<'a>(alpha: f64, inplace: bool) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_elu(alpha, inplace).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_silu<'a>() -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_silu().map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_mish<'a>() -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_mish().map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_prelu<'a>(num_parameters: i64) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_prelu(num_parameters).map_err(cxx_err)?)
}

#[rustler::nif]
pub fn nn_log_softmax<'a>(dim: i64) -> NifResult<NNModuleStruct<'a>> {
    wrap_nn_module(torch::nn_log_softmax(dim).map_err(cxx_err)?)
}

// Module operations

#[rustler::nif]
pub fn nn_forward<'a>(module: NNModuleStruct<'a>, input: TensorStruct<'a>) -> NifResult<TensorStruct<'a>> {
    let result = torch::nn_forward(&module.resource.module, &input.resource.tensor).map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn nn_parameters<'a>(env: Env<'a>, module: NNModuleStruct<'a>) -> NifResult<Term<'a>> {
    let params = torch::nn_parameters(&module.resource.module).map_err(cxx_err)?;
    Ok(named_tensors_to_term(env, &params))
}

#[rustler::nif]
pub fn nn_set_eval(module: NNModuleStruct) -> NifResult<()> {
    torch::nn_set_eval(&module.resource.module).map_err(cxx_err)
}

#[rustler::nif]
pub fn nn_set_train(module: NNModuleStruct) -> NifResult<()> {
    torch::nn_set_train(&module.resource.module).map_err(cxx_err)
}

#[rustler::nif]
pub fn nn_type_name(module: NNModuleStruct) -> NifResult<String> {
    let name = torch::nn_type_name(&module.resource.module).map_err(cxx_err)?;
    Ok(name.to_string())
}

/// Copy parameter values from a list of {name, tensor} into a module's parameters.
#[rustler::nif]
pub fn nn_copy_parameters<'a>(dst: NNModuleStruct<'a>, params: Vec<(String, TensorStruct<'a>)>) -> NifResult<()> {
    let named: Vec<torch::NamedTensor> = params
        .iter()
        .map(|(name, t)| torch::NamedTensor {
            name: name.clone(),
            tensor: t.resource.tensor.clone(),
        })
        .collect();

    torch::nn_copy_parameters(&dst.resource.module, named).map_err(cxx_err)
}

#[rustler::nif]
pub fn nn_to_device<'a>(module: NNModuleStruct<'a>, device: torch::Device) -> NifResult<NNModuleStruct<'a>> {
    let result = torch::nn_to_device(&module.resource.module, device).map_err(cxx_err)?;
    wrap_nn_module(result)
}
