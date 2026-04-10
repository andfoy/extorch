use crate::encoding::jit::named_tensors_to_term;
use crate::native::torch;
use crate::shared_types::{NNModuleStruct, Reference, TensorStruct};

use rustler::{Encoder, Env, Error, NifResult, ResourceArc, Term};

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

// ============================================================================
// Direct ATen functional ops (Phase A)
// ============================================================================

fn pack_inputs<'a>(tensors: &[TensorStruct<'a>]) -> torch::TensorList {
    let values: Vec<torch::TensorOut> = tensors
        .iter()
        .map(|t| torch::TensorOut {
            tensor: t.resource.tensor.clone(),
            used: true,
        })
        .collect();
    torch::TensorList { values, used: true }
}

fn unpack_outputs<'a>(env: Env<'a>, list: torch::TensorList) -> Term<'a> {
    let out: Vec<Term<'a>> = list
        .values
        .iter()
        .filter(|t| t.used)
        .map(|t| {
            let ts: TensorStruct<'a> = t.tensor.clone().into();
            ts.encode(env)
        })
        .collect();
    out.encode(env)
}

#[rustler::nif]
pub fn aten_lstm<'a>(
    env: Env<'a>,
    input: TensorStruct<'a>,
    hx: Vec<TensorStruct<'a>>,
    params: Vec<TensorStruct<'a>>,
    has_biases: bool,
    num_layers: i64,
    dropout: f64,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> NifResult<Term<'a>> {
    let hx_list = pack_inputs(&hx);
    let params_list = pack_inputs(&params);
    let result = torch::aten_lstm(
        &input.resource.tensor,
        hx_list,
        params_list,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )
    .map_err(cxx_err)?;
    Ok(unpack_outputs(env, result))
}

#[rustler::nif]
pub fn aten_gru<'a>(
    env: Env<'a>,
    input: TensorStruct<'a>,
    hx: TensorStruct<'a>,
    params: Vec<TensorStruct<'a>>,
    has_biases: bool,
    num_layers: i64,
    dropout: f64,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> NifResult<Term<'a>> {
    let params_list = pack_inputs(&params);
    let result = torch::aten_gru(
        &input.resource.tensor,
        &hx.resource.tensor,
        params_list,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )
    .map_err(cxx_err)?;
    Ok(unpack_outputs(env, result))
}

#[rustler::nif]
pub fn aten_transformer_encoder_layer_fwd<'a>(
    src: TensorStruct<'a>,
    embed_dim: i64,
    num_heads: i64,
    qkv_weight: TensorStruct<'a>,
    qkv_bias: TensorStruct<'a>,
    proj_weight: TensorStruct<'a>,
    proj_bias: TensorStruct<'a>,
    use_gelu: bool,
    norm_first: bool,
    eps: f64,
    norm_weight_1: TensorStruct<'a>,
    norm_bias_1: TensorStruct<'a>,
    norm_weight_2: TensorStruct<'a>,
    norm_bias_2: TensorStruct<'a>,
    ffn_weight_1: TensorStruct<'a>,
    ffn_bias_1: TensorStruct<'a>,
    ffn_weight_2: TensorStruct<'a>,
    ffn_bias_2: TensorStruct<'a>,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_transformer_encoder_layer_fwd(
        &src.resource.tensor,
        embed_dim,
        num_heads,
        &qkv_weight.resource.tensor,
        &qkv_bias.resource.tensor,
        &proj_weight.resource.tensor,
        &proj_bias.resource.tensor,
        use_gelu,
        norm_first,
        eps,
        &norm_weight_1.resource.tensor,
        &norm_bias_1.resource.tensor,
        &norm_weight_2.resource.tensor,
        &norm_bias_2.resource.tensor,
        &ffn_weight_1.resource.tensor,
        &ffn_bias_1.resource.tensor,
        &ffn_weight_2.resource.tensor,
        &ffn_bias_2.resource.tensor,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_scaled_dot_product_attention<'a>(
    query: TensorStruct<'a>,
    key: TensorStruct<'a>,
    value: TensorStruct<'a>,
    dropout_p: f64,
    is_causal: bool,
    scale: f64,
    has_scale: bool,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_scaled_dot_product_attention(
        &query.resource.tensor,
        &key.resource.tensor,
        &value.resource.tensor,
        dropout_p,
        is_causal,
        scale,
        has_scale,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_native_multi_head_attention<'a>(
    env: Env<'a>,
    query: TensorStruct<'a>,
    key: TensorStruct<'a>,
    value: TensorStruct<'a>,
    embed_dim: i64,
    num_head: i64,
    qkv_weight: TensorStruct<'a>,
    qkv_bias: TensorStruct<'a>,
    proj_weight: TensorStruct<'a>,
    proj_bias: TensorStruct<'a>,
    need_weights: bool,
    average_attn_weights: bool,
) -> NifResult<Term<'a>> {
    let result = torch::aten_native_multi_head_attention(
        &query.resource.tensor,
        &key.resource.tensor,
        &value.resource.tensor,
        embed_dim,
        num_head,
        &qkv_weight.resource.tensor,
        &qkv_bias.resource.tensor,
        &proj_weight.resource.tensor,
        &proj_bias.resource.tensor,
        need_weights,
        average_attn_weights,
    )
    .map_err(cxx_err)?;
    Ok(unpack_outputs(env, result))
}

// ============================================================================
// Direct ATen functional ops (Phase B): conv, norm, pool
// ============================================================================

// Pack a 0-or-1-element optional tensor into a TensorList for the cxx bridge.
fn pack_optional<'a>(opt: &Option<TensorStruct<'a>>) -> torch::TensorList {
    let values: Vec<torch::TensorOut> = opt
        .iter()
        .map(|t| torch::TensorOut {
            tensor: t.resource.tensor.clone(),
            used: true,
        })
        .collect();
    torch::TensorList { values, used: true }
}

#[rustler::nif]
pub fn aten_conv2d<'a>(
    input: TensorStruct<'a>,
    weight: TensorStruct<'a>,
    bias_opt: Option<TensorStruct<'a>>,
    stride: Vec<i64>,
    padding: Vec<i64>,
    dilation: Vec<i64>,
    groups: i64,
) -> NifResult<TensorStruct<'a>> {
    let bias_list = pack_optional(&bias_opt);
    let result = torch::aten_conv2d(
        &input.resource.tensor,
        &weight.resource.tensor,
        bias_list,
        stride,
        padding,
        dilation,
        groups,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_conv1d<'a>(
    input: TensorStruct<'a>,
    weight: TensorStruct<'a>,
    bias_opt: Option<TensorStruct<'a>>,
    stride: Vec<i64>,
    padding: Vec<i64>,
    dilation: Vec<i64>,
    groups: i64,
) -> NifResult<TensorStruct<'a>> {
    let bias_list = pack_optional(&bias_opt);
    let result = torch::aten_conv1d(
        &input.resource.tensor,
        &weight.resource.tensor,
        bias_list,
        stride,
        padding,
        dilation,
        groups,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_conv3d<'a>(
    input: TensorStruct<'a>,
    weight: TensorStruct<'a>,
    bias_opt: Option<TensorStruct<'a>>,
    stride: Vec<i64>,
    padding: Vec<i64>,
    dilation: Vec<i64>,
    groups: i64,
) -> NifResult<TensorStruct<'a>> {
    let bias_list = pack_optional(&bias_opt);
    let result = torch::aten_conv3d(
        &input.resource.tensor,
        &weight.resource.tensor,
        bias_list,
        stride,
        padding,
        dilation,
        groups,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_convolution<'a>(
    input: TensorStruct<'a>,
    weight: TensorStruct<'a>,
    bias_opt: Option<TensorStruct<'a>>,
    stride: Vec<i64>,
    padding: Vec<i64>,
    dilation: Vec<i64>,
    transposed: bool,
    output_padding: Vec<i64>,
    groups: i64,
) -> NifResult<TensorStruct<'a>> {
    let bias_list = pack_optional(&bias_opt);
    let result = torch::aten_convolution(
        &input.resource.tensor,
        &weight.resource.tensor,
        bias_list,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_conv_transpose2d<'a>(
    input: TensorStruct<'a>,
    weight: TensorStruct<'a>,
    bias_opt: Option<TensorStruct<'a>>,
    stride: Vec<i64>,
    padding: Vec<i64>,
    output_padding: Vec<i64>,
    groups: i64,
    dilation: Vec<i64>,
) -> NifResult<TensorStruct<'a>> {
    let bias_list = pack_optional(&bias_opt);
    let result = torch::aten_conv_transpose2d(
        &input.resource.tensor,
        &weight.resource.tensor,
        bias_list,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_batch_norm<'a>(
    input: TensorStruct<'a>,
    weight_opt: Option<TensorStruct<'a>>,
    bias_opt: Option<TensorStruct<'a>>,
    running_mean_opt: Option<TensorStruct<'a>>,
    running_var_opt: Option<TensorStruct<'a>>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_batch_norm(
        &input.resource.tensor,
        pack_optional(&weight_opt),
        pack_optional(&bias_opt),
        pack_optional(&running_mean_opt),
        pack_optional(&running_var_opt),
        training,
        momentum,
        eps,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_layer_norm<'a>(
    input: TensorStruct<'a>,
    normalized_shape: Vec<i64>,
    weight_opt: Option<TensorStruct<'a>>,
    bias_opt: Option<TensorStruct<'a>>,
    eps: f64,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_layer_norm(
        &input.resource.tensor,
        normalized_shape,
        pack_optional(&weight_opt),
        pack_optional(&bias_opt),
        eps,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_group_norm<'a>(
    input: TensorStruct<'a>,
    num_groups: i64,
    weight_opt: Option<TensorStruct<'a>>,
    bias_opt: Option<TensorStruct<'a>>,
    eps: f64,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_group_norm(
        &input.resource.tensor,
        num_groups,
        pack_optional(&weight_opt),
        pack_optional(&bias_opt),
        eps,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_max_pool2d<'a>(
    input: TensorStruct<'a>,
    kernel_size: Vec<i64>,
    stride: Vec<i64>,
    padding: Vec<i64>,
    dilation: Vec<i64>,
    ceil_mode: bool,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_max_pool2d(
        &input.resource.tensor,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_avg_pool2d<'a>(
    input: TensorStruct<'a>,
    kernel_size: Vec<i64>,
    stride: Vec<i64>,
    padding: Vec<i64>,
    ceil_mode: bool,
    count_include_pad: bool,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_avg_pool2d(
        &input.resource.tensor,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_adaptive_avg_pool2d<'a>(
    input: TensorStruct<'a>,
    output_size: Vec<i64>,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_adaptive_avg_pool2d(
        &input.resource.tensor,
        output_size,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

// Thread pool / backend inspection

#[rustler::nif]
pub fn aten_get_num_threads() -> NifResult<i64> {
    torch::aten_get_num_threads().map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_set_num_threads(n: i64) -> NifResult<()> {
    torch::aten_set_num_threads(n).map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_get_num_interop_threads() -> NifResult<i64> {
    torch::aten_get_num_interop_threads().map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_set_num_interop_threads(n: i64) -> NifResult<()> {
    torch::aten_set_num_interop_threads(n).map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_mkldnn_is_available() -> NifResult<bool> {
    torch::aten_mkldnn_is_available().map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_noop<'a>(t: TensorStruct<'a>) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_noop(&t.resource.tensor).map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_backend_info() -> NifResult<String> {
    Ok(torch::aten_backend_info().map_err(cxx_err)?.to_string())
}

#[rustler::nif]
pub fn aten_is_grad_enabled() -> NifResult<bool> {
    torch::aten_is_grad_enabled().map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_set_grad_enabled(enabled: bool) -> NifResult<()> {
    torch::aten_set_grad_enabled(enabled).map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_clear_cpu_affinity() -> NifResult<bool> {
    torch::aten_clear_cpu_affinity().map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_cuda_synchronize() -> NifResult<()> {
    torch::aten_cuda_synchronize().map_err(cxx_err)
}

#[rustler::nif]
pub fn aten_mkldnn_reorder_conv2d_weight<'a>(
    weight: TensorStruct<'a>,
    padding: Vec<i64>,
    stride: Vec<i64>,
    dilation: Vec<i64>,
    groups: i64,
) -> NifResult<TensorStruct<'a>> {
    let result = torch::aten_mkldnn_reorder_conv2d_weight(
        &weight.resource.tensor,
        padding,
        stride,
        dilation,
        groups,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}

#[rustler::nif]
pub fn aten_mkldnn_convolution<'a>(
    input: TensorStruct<'a>,
    packed_weight: TensorStruct<'a>,
    bias_opt: Option<TensorStruct<'a>>,
    padding: Vec<i64>,
    stride: Vec<i64>,
    dilation: Vec<i64>,
    groups: i64,
) -> NifResult<TensorStruct<'a>> {
    let bias_list = pack_optional(&bias_opt);
    let result = torch::aten_mkldnn_convolution(
        &input.resource.tensor,
        &packed_weight.resource.tensor,
        bias_list,
        padding,
        stride,
        dilation,
        groups,
    )
    .map_err(cxx_err)?;
    Ok(result.into())
}
