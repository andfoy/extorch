#include "extorch/src/native.rs.h"
#include "extorch/include/nn.h"

#ifdef __linux__
#define _GNU_SOURCE
#include <sched.h>
#include <thread>
#endif


// Helper: convert Device struct to torch::Device
static torch::Device make_device(const Device &s_device) {
    std::string device_str(s_device.device);
    auto it = device_mapping.find(device_str);
    if (it == device_mapping.end()) {
        throw std::runtime_error("Unknown device: " + device_str);
    }
    if (s_device.index >= 0) {
        return torch::Device(it->second, s_device.index);
    }
    return torch::Device(it->second);
}

// ============================================================================
// Layer factory functions
// ============================================================================

std::shared_ptr<CrossNNModule> nn_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias)
{
    auto opts = torch::nn::LinearOptions(in_features, out_features).bias(bias);
    auto module = torch::nn::Linear(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "Linear");
}

std::shared_ptr<CrossNNModule> nn_conv1d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool bias)
{
    auto opts = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias);
    auto module = torch::nn::Conv1d(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "Conv1d");
}

std::shared_ptr<CrossNNModule> nn_conv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool bias)
{
    auto opts = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias);
    auto module = torch::nn::Conv2d(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "Conv2d");
}

std::shared_ptr<CrossNNModule> nn_batch_norm1d(
    int64_t num_features,
    double eps,
    double momentum,
    bool affine,
    bool track_running_stats)
{
    auto opts = torch::nn::BatchNorm1dOptions(num_features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);
    auto module = torch::nn::BatchNorm1d(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "BatchNorm1d");
}

std::shared_ptr<CrossNNModule> nn_batch_norm2d(
    int64_t num_features,
    double eps,
    double momentum,
    bool affine,
    bool track_running_stats)
{
    auto opts = torch::nn::BatchNorm2dOptions(num_features)
        .eps(eps)
        .momentum(momentum)
        .affine(affine)
        .track_running_stats(track_running_stats);
    auto module = torch::nn::BatchNorm2d(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "BatchNorm2d");
}

std::shared_ptr<CrossNNModule> nn_layer_norm(
    rust::Vec<int64_t> normalized_shape,
    double eps,
    bool elementwise_affine)
{
    std::vector<int64_t> shape(normalized_shape.begin(), normalized_shape.end());
    auto opts = torch::nn::LayerNormOptions(shape)
        .eps(eps)
        .elementwise_affine(elementwise_affine);
    auto module = torch::nn::LayerNorm(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "LayerNorm");
}

std::shared_ptr<CrossNNModule> nn_dropout(double p, bool inplace)
{
    auto opts = torch::nn::DropoutOptions().p(p).inplace(inplace);
    auto module = torch::nn::Dropout(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "Dropout");
}

std::shared_ptr<CrossNNModule> nn_embedding(
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t padding_idx,
    bool has_padding_idx)
{
    auto opts = torch::nn::EmbeddingOptions(num_embeddings, embedding_dim);
    if (has_padding_idx) {
        opts.padding_idx(padding_idx);
    }
    auto module = torch::nn::Embedding(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "Embedding");
}

// ============================================================================
// Activation functions
// ============================================================================

std::shared_ptr<CrossNNModule> nn_relu(bool inplace)
{
    auto opts = torch::nn::ReLUOptions().inplace(inplace);
    auto module = torch::nn::ReLU(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "ReLU");
}

std::shared_ptr<CrossNNModule> nn_gelu()
{
    auto module = torch::nn::GELU();
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "GELU");
}

std::shared_ptr<CrossNNModule> nn_sigmoid()
{
    auto module = torch::nn::Sigmoid();
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "Sigmoid");
}

std::shared_ptr<CrossNNModule> nn_tanh()
{
    auto module = torch::nn::Tanh();
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "Tanh");
}

std::shared_ptr<CrossNNModule> nn_softmax(int64_t dim)
{
    auto opts = torch::nn::SoftmaxOptions(dim);
    auto module = torch::nn::Softmax(opts);
    return std::make_shared<CrossNNModule>(
        torch::nn::AnyModule(module), "Softmax");
}

std::shared_ptr<CrossNNModule> nn_leaky_relu(double negative_slope, bool inplace) {
    auto opts = torch::nn::LeakyReLUOptions().negative_slope(negative_slope).inplace(inplace);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::LeakyReLU(opts)), "LeakyReLU");
}

std::shared_ptr<CrossNNModule> nn_elu(double alpha, bool inplace) {
    auto opts = torch::nn::ELUOptions().alpha(alpha).inplace(inplace);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::ELU(opts)), "ELU");
}

std::shared_ptr<CrossNNModule> nn_silu() {
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::SiLU()), "SiLU");
}

std::shared_ptr<CrossNNModule> nn_mish() {
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::Mish()), "Mish");
}

std::shared_ptr<CrossNNModule> nn_prelu(int64_t num_parameters) {
    auto opts = torch::nn::PReLUOptions().num_parameters(num_parameters);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::PReLU(opts)), "PReLU");
}

std::shared_ptr<CrossNNModule> nn_log_softmax(int64_t dim) {
    auto opts = torch::nn::LogSoftmaxOptions(dim);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::LogSoftmax(opts)), "LogSoftmax");
}

// ============================================================================
// Additional convolutions
// ============================================================================

std::shared_ptr<CrossNNModule> nn_conv3d(
    int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool bias) {
    auto opts = torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
        .stride(stride).padding(padding).dilation(dilation).groups(groups).bias(bias);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::Conv3d(opts)), "Conv3d");
}

std::shared_ptr<CrossNNModule> nn_conv_transpose1d(
    int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t output_padding, int64_t groups, bool bias, int64_t dilation) {
    auto opts = torch::nn::ConvTranspose1dOptions(in_channels, out_channels, kernel_size)
        .stride(stride).padding(padding).output_padding(output_padding).groups(groups).bias(bias).dilation(dilation);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::ConvTranspose1d(opts)), "ConvTranspose1d");
}

std::shared_ptr<CrossNNModule> nn_conv_transpose2d(
    int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t output_padding, int64_t groups, bool bias, int64_t dilation) {
    auto opts = torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size)
        .stride(stride).padding(padding).output_padding(output_padding).groups(groups).bias(bias).dilation(dilation);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::ConvTranspose2d(opts)), "ConvTranspose2d");
}

// ============================================================================
// Pooling
// ============================================================================

std::shared_ptr<CrossNNModule> nn_max_pool1d(int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode) {
    auto opts = torch::nn::MaxPool1dOptions(kernel_size).stride(stride).padding(padding).dilation(dilation).ceil_mode(ceil_mode);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::MaxPool1d(opts)), "MaxPool1d");
}

std::shared_ptr<CrossNNModule> nn_max_pool2d(int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode) {
    auto opts = torch::nn::MaxPool2dOptions(kernel_size).stride(stride).padding(padding).dilation(dilation).ceil_mode(ceil_mode);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::MaxPool2d(opts)), "MaxPool2d");
}

std::shared_ptr<CrossNNModule> nn_avg_pool1d(int64_t kernel_size, int64_t stride, int64_t padding, bool ceil_mode, bool count_include_pad) {
    auto opts = torch::nn::AvgPool1dOptions(kernel_size).stride(stride).padding(padding).ceil_mode(ceil_mode).count_include_pad(count_include_pad);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::AvgPool1d(opts)), "AvgPool1d");
}

std::shared_ptr<CrossNNModule> nn_avg_pool2d(int64_t kernel_size, int64_t stride, int64_t padding, bool ceil_mode, bool count_include_pad) {
    auto opts = torch::nn::AvgPool2dOptions(kernel_size).stride(stride).padding(padding).ceil_mode(ceil_mode).count_include_pad(count_include_pad);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::AvgPool2d(opts)), "AvgPool2d");
}

std::shared_ptr<CrossNNModule> nn_adaptive_avg_pool1d(int64_t output_size) {
    auto opts = torch::nn::AdaptiveAvgPool1dOptions(output_size);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::AdaptiveAvgPool1d(opts)), "AdaptiveAvgPool1d");
}

std::shared_ptr<CrossNNModule> nn_adaptive_avg_pool2d(int64_t output_h, int64_t output_w) {
    auto opts = torch::nn::AdaptiveAvgPool2dOptions({output_h, output_w});
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::AdaptiveAvgPool2d(opts)), "AdaptiveAvgPool2d");
}

// ============================================================================
// Additional normalization
// ============================================================================

std::shared_ptr<CrossNNModule> nn_group_norm(int64_t num_groups, int64_t num_channels, double eps, bool affine) {
    auto opts = torch::nn::GroupNormOptions(num_groups, num_channels).eps(eps).affine(affine);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::GroupNorm(opts)), "GroupNorm");
}

std::shared_ptr<CrossNNModule> nn_instance_norm1d(int64_t num_features, double eps, double momentum, bool affine, bool track_running_stats) {
    auto opts = torch::nn::InstanceNorm1dOptions(num_features).eps(eps).momentum(momentum).affine(affine).track_running_stats(track_running_stats);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::InstanceNorm1d(opts)), "InstanceNorm1d");
}

std::shared_ptr<CrossNNModule> nn_instance_norm2d(int64_t num_features, double eps, double momentum, bool affine, bool track_running_stats) {
    auto opts = torch::nn::InstanceNorm2dOptions(num_features).eps(eps).momentum(momentum).affine(affine).track_running_stats(track_running_stats);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::InstanceNorm2d(opts)), "InstanceNorm2d");
}

// ============================================================================
// Recurrent
// ============================================================================

std::shared_ptr<CrossNNModule> nn_lstm(int64_t input_size, int64_t hidden_size, int64_t num_layers,
    bool bias, bool batch_first, double dropout, bool bidirectional) {
    auto opts = torch::nn::LSTMOptions(input_size, hidden_size)
        .num_layers(num_layers).bias(bias).batch_first(batch_first).dropout(dropout).bidirectional(bidirectional);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::LSTM(opts)), "LSTM");
}

std::shared_ptr<CrossNNModule> nn_gru(int64_t input_size, int64_t hidden_size, int64_t num_layers,
    bool bias, bool batch_first, double dropout, bool bidirectional) {
    auto opts = torch::nn::GRUOptions(input_size, hidden_size)
        .num_layers(num_layers).bias(bias).batch_first(batch_first).dropout(dropout).bidirectional(bidirectional);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::GRU(opts)), "GRU");
}

// ============================================================================
// Transformer
// ============================================================================

std::shared_ptr<CrossNNModule> nn_multihead_attention(int64_t embed_dim, int64_t num_heads, double dropout, bool bias) {
    auto opts = torch::nn::MultiheadAttentionOptions(embed_dim, num_heads).dropout(dropout).bias(bias);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::MultiheadAttention(opts)), "MultiheadAttention");
}

// ============================================================================
// Utility
// ============================================================================

std::shared_ptr<CrossNNModule> nn_flatten(int64_t start_dim, int64_t end_dim) {
    auto opts = torch::nn::FlattenOptions().start_dim(start_dim).end_dim(end_dim);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::Flatten(opts)), "Flatten");
}

std::shared_ptr<CrossNNModule> nn_unflatten(int64_t dim, rust::Vec<int64_t> sizes) {
    std::vector<int64_t> shape(sizes.begin(), sizes.end());
    auto opts = torch::nn::UnflattenOptions(dim, shape);
    return std::make_shared<CrossNNModule>(torch::nn::AnyModule(torch::nn::Unflatten(opts)), "Unflatten");
}

// ============================================================================
// Module operations
// ============================================================================

std::shared_ptr<CrossTensor> nn_forward(
    const std::shared_ptr<CrossNNModule> &module,
    const std::shared_ptr<CrossTensor> &input)
{
    auto result = module->module.forward(*input);
    return std::make_shared<CrossTensor>(std::move(result));
}

rust::Vec<NamedTensor> nn_parameters(
    const std::shared_ptr<CrossNNModule> &module)
{
    rust::Vec<NamedTensor> params;
    auto base_module = module->module.ptr();
    for (const auto &param : base_module->named_parameters()) {
        NamedTensor nt;
        nt.name = rust::String(param.key());
        nt.tensor = std::make_shared<CrossTensor>(param.value());
        params.push_back(std::move(nt));
    }
    return params;
}

void nn_set_eval(const std::shared_ptr<CrossNNModule> &module) {
    module->module.ptr()->eval();
}

void nn_set_train(const std::shared_ptr<CrossNNModule> &module) {
    module->module.ptr()->train();
}

rust::String nn_type_name(const std::shared_ptr<CrossNNModule> &module) {
    return rust::String(module->type_name);
}

void nn_copy_parameters(
    const std::shared_ptr<CrossNNModule> &dst,
    rust::Vec<NamedTensor> params)
{
    auto base = dst->module.ptr();
    auto param_dict = base->named_parameters();

    for (const auto &nt : params) {
        std::string name(nt.name);
        auto it = param_dict.find(name);
        if (it != nullptr) {
            // Copy data from source tensor into existing parameter (no grad)
            torch::NoGradGuard no_grad;
            it->copy_(*nt.tensor);
        }
    }
}

std::shared_ptr<CrossNNModule> nn_to_device(
    const std::shared_ptr<CrossNNModule> &module,
    Device s_device)
{
    auto device = make_device(s_device);
    // Clone the underlying module and move to device
    auto cloned = module->module.clone();
    cloned.ptr()->to(device);
    return std::make_shared<CrossNNModule>(
        std::move(cloned), module->type_name);
}

// ============================================================================
// Direct ATen functional ops (Phase A)
// ============================================================================

TensorList aten_lstm(
    const std::shared_ptr<CrossTensor> &input,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first)
{
    auto hx_vec = unpack_tensor_list(std::move(hx));
    auto params_vec = unpack_tensor_list(std::move(params));

    auto result = at::lstm(
        *input,
        at::TensorList(hx_vec),
        at::TensorList(params_vec),
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first);

    std::vector<CrossTensor> outputs;
    outputs.push_back(std::get<0>(result));
    outputs.push_back(std::get<1>(result));
    outputs.push_back(std::get<2>(result));
    return pack_tensor_list(std::move(outputs));
}

TensorList aten_gru(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first)
{
    auto params_vec = unpack_tensor_list(std::move(params));

    auto result = at::gru(
        *input,
        *hx,
        at::TensorList(params_vec),
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first);

    std::vector<CrossTensor> outputs;
    outputs.push_back(std::get<0>(result));
    outputs.push_back(std::get<1>(result));
    return pack_tensor_list(std::move(outputs));
}

std::shared_ptr<CrossTensor> aten_transformer_encoder_layer_fwd(
    const std::shared_ptr<CrossTensor> &src,
    int64_t embed_dim,
    int64_t num_heads,
    const std::shared_ptr<CrossTensor> &qkv_weight,
    const std::shared_ptr<CrossTensor> &qkv_bias,
    const std::shared_ptr<CrossTensor> &proj_weight,
    const std::shared_ptr<CrossTensor> &proj_bias,
    bool use_gelu,
    bool norm_first,
    double eps,
    const std::shared_ptr<CrossTensor> &norm_weight_1,
    const std::shared_ptr<CrossTensor> &norm_bias_1,
    const std::shared_ptr<CrossTensor> &norm_weight_2,
    const std::shared_ptr<CrossTensor> &norm_bias_2,
    const std::shared_ptr<CrossTensor> &ffn_weight_1,
    const std::shared_ptr<CrossTensor> &ffn_bias_1,
    const std::shared_ptr<CrossTensor> &ffn_weight_2,
    const std::shared_ptr<CrossTensor> &ffn_bias_2)
{
    auto result = at::_transformer_encoder_layer_fwd(
        *src,
        embed_dim,
        num_heads,
        *qkv_weight,
        *qkv_bias,
        *proj_weight,
        *proj_bias,
        use_gelu,
        norm_first,
        eps,
        *norm_weight_1,
        *norm_bias_1,
        *norm_weight_2,
        *norm_bias_2,
        *ffn_weight_1,
        *ffn_bias_1,
        *ffn_weight_2,
        *ffn_bias_2,
        c10::nullopt,  // mask
        c10::nullopt); // mask_type
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_scaled_dot_product_attention(
    const std::shared_ptr<CrossTensor> &query,
    const std::shared_ptr<CrossTensor> &key,
    const std::shared_ptr<CrossTensor> &value,
    double dropout_p,
    bool is_causal,
    double scale,
    bool has_scale)
{
    c10::optional<double> scale_opt;
    if (has_scale) scale_opt = scale;
    auto result = at::scaled_dot_product_attention(
        *query, *key, *value,
        c10::nullopt,  // attn_mask
        dropout_p,
        is_causal,
        scale_opt);
    return std::make_shared<CrossTensor>(std::move(result));
}

TensorList aten_native_multi_head_attention(
    const std::shared_ptr<CrossTensor> &query,
    const std::shared_ptr<CrossTensor> &key,
    const std::shared_ptr<CrossTensor> &value,
    int64_t embed_dim,
    int64_t num_head,
    const std::shared_ptr<CrossTensor> &qkv_weight,
    const std::shared_ptr<CrossTensor> &qkv_bias,
    const std::shared_ptr<CrossTensor> &proj_weight,
    const std::shared_ptr<CrossTensor> &proj_bias,
    bool need_weights,
    bool average_attn_weights)
{
    auto result = at::_native_multi_head_attention(
        *query, *key, *value,
        embed_dim,
        num_head,
        *qkv_weight,
        *qkv_bias,
        *proj_weight,
        *proj_bias,
        c10::nullopt,  // mask
        need_weights,
        average_attn_weights,
        c10::nullopt); // mask_type

    std::vector<CrossTensor> outputs;
    outputs.push_back(std::get<0>(result));
    outputs.push_back(std::get<1>(result));
    return pack_tensor_list(std::move(outputs));
}

// ============================================================================
// Direct ATen functional ops (Phase B)
// ============================================================================

// Helper: extract optional tensor from a TensorList of 0 or 1 elements.
static c10::optional<at::Tensor> opt_tensor(const TensorList &list) {
    if (list.values.empty() || !list.values[0].used) return c10::nullopt;
    return *list.values[0].tensor;
}

std::shared_ptr<CrossTensor> aten_conv2d(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    int64_t groups)
{
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    std::vector<int64_t> dilation_v(dilation.begin(), dilation.end());
    auto bias = opt_tensor(bias_opt);
    auto result = at::conv2d(*input, *weight, bias,
        stride_v, padding_v, dilation_v, groups);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_conv1d(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    int64_t groups)
{
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    std::vector<int64_t> dilation_v(dilation.begin(), dilation.end());
    auto bias = opt_tensor(bias_opt);
    auto result = at::conv1d(*input, *weight, bias,
        stride_v, padding_v, dilation_v, groups);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_conv3d(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    int64_t groups)
{
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    std::vector<int64_t> dilation_v(dilation.begin(), dilation.end());
    auto bias = opt_tensor(bias_opt);
    auto result = at::conv3d(*input, *weight, bias,
        stride_v, padding_v, dilation_v, groups);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_convolution(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    bool transposed,
    rust::Vec<int64_t> output_padding,
    int64_t groups)
{
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    std::vector<int64_t> dilation_v(dilation.begin(), dilation.end());
    std::vector<int64_t> output_padding_v(output_padding.begin(), output_padding.end());
    auto bias = opt_tensor(bias_opt);
    auto result = at::convolution(*input, *weight, bias,
        stride_v, padding_v, dilation_v,
        transposed, output_padding_v, groups);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_conv_transpose2d(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> output_padding,
    int64_t groups,
    rust::Vec<int64_t> dilation)
{
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    std::vector<int64_t> output_padding_v(output_padding.begin(), output_padding.end());
    std::vector<int64_t> dilation_v(dilation.begin(), dilation.end());
    auto bias = opt_tensor(bias_opt);
    auto result = at::conv_transpose2d(*input, *weight, bias,
        stride_v, padding_v, output_padding_v, groups, dilation_v);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_batch_norm(
    const std::shared_ptr<CrossTensor> &input,
    TensorList weight_opt,
    TensorList bias_opt,
    TensorList running_mean_opt,
    TensorList running_var_opt,
    bool training,
    double momentum,
    double eps)
{
    auto weight = opt_tensor(weight_opt);
    auto bias = opt_tensor(bias_opt);
    auto running_mean = opt_tensor(running_mean_opt);
    auto running_var = opt_tensor(running_var_opt);
    auto result = at::batch_norm(*input,
        weight, bias, running_mean, running_var,
        training, momentum, eps,
        /*cudnn_enabled=*/true);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_layer_norm(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> normalized_shape,
    TensorList weight_opt,
    TensorList bias_opt,
    double eps)
{
    std::vector<int64_t> shape(normalized_shape.begin(), normalized_shape.end());
    auto weight = opt_tensor(weight_opt);
    auto bias = opt_tensor(bias_opt);
    auto result = at::layer_norm(*input, shape, weight, bias, eps,
        /*cudnn_enable=*/true);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_group_norm(
    const std::shared_ptr<CrossTensor> &input,
    int64_t num_groups,
    TensorList weight_opt,
    TensorList bias_opt,
    double eps)
{
    auto weight = opt_tensor(weight_opt);
    auto bias = opt_tensor(bias_opt);
    auto result = at::group_norm(*input, num_groups, weight, bias, eps,
        /*cudnn_enabled=*/true);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_max_pool2d(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> kernel_size,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    bool ceil_mode)
{
    std::vector<int64_t> kernel_v(kernel_size.begin(), kernel_size.end());
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    std::vector<int64_t> dilation_v(dilation.begin(), dilation.end());
    auto result = at::max_pool2d(*input, kernel_v, stride_v, padding_v, dilation_v, ceil_mode);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_avg_pool2d(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> kernel_size,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    bool ceil_mode,
    bool count_include_pad)
{
    std::vector<int64_t> kernel_v(kernel_size.begin(), kernel_size.end());
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    auto result = at::avg_pool2d(*input, kernel_v, stride_v, padding_v,
        ceil_mode, count_include_pad, c10::nullopt);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_adaptive_avg_pool2d(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> output_size)
{
    std::vector<int64_t> out_v(output_size.begin(), output_size.end());
    auto result = at::adaptive_avg_pool2d(*input, out_v);
    return std::make_shared<CrossTensor>(std::move(result));
}

// ============================================================================
// Thread pool / backend inspection
// ============================================================================

int64_t aten_get_num_threads() {
    return static_cast<int64_t>(at::get_num_threads());
}

void aten_set_num_threads(int64_t n) {
    at::set_num_threads(static_cast<int>(n));
}

int64_t aten_get_num_interop_threads() {
    return static_cast<int64_t>(at::get_num_interop_threads());
}

void aten_set_num_interop_threads(int64_t n) {
    at::set_num_interop_threads(static_cast<int>(n));
}

bool aten_mkldnn_is_available() {
#if AT_MKLDNN_ENABLED()
    return true;
#else
    return false;
#endif
}

std::shared_ptr<CrossTensor> aten_noop(const std::shared_ptr<CrossTensor> &t) {
    return t;
}

rust::String aten_backend_info() {
    auto &ctx = at::globalContext();
    std::string s;
    s += "userEnabledMkldnn=" + std::to_string(ctx.userEnabledMkldnn()) + " ";
    s += "userEnabledNNPACK=" + std::to_string(ctx.userEnabledNNPACK()) + " ";
    s += "benchmarkCuDNN=" + std::to_string(ctx.benchmarkCuDNN()) + " ";
    s += "deterministicCuDNN=" + std::to_string(ctx.deterministicCuDNN()) + " ";
    s += "allowTF32CuBLAS=" + std::to_string(ctx.allowTF32CuBLAS()) + " ";
    s += "intra_op_parallelism_threads=" + std::to_string(at::get_num_threads()) + " ";
    s += "inter_op_parallelism_threads=" + std::to_string(at::get_num_interop_threads()) + " ";
    s += "grad_enabled=" + std::to_string(torch::GradMode::is_enabled());
    return rust::String(s);
}

bool aten_is_grad_enabled() {
    return torch::GradMode::is_enabled();
}

void aten_set_grad_enabled(bool enabled) {
    torch::GradMode::set_enabled(enabled);
}

bool aten_clear_cpu_affinity() {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int ncpu = static_cast<int>(std::thread::hardware_concurrency());
    for (int i = 0; i < ncpu; i++) CPU_SET(i, &mask);
    return sched_setaffinity(0, sizeof(mask), &mask) == 0;
#else
    return false;
#endif
}

std::shared_ptr<CrossTensor> aten_mkldnn_reorder_conv2d_weight(
    const std::shared_ptr<CrossTensor> &weight,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> dilation,
    int64_t groups)
{
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> dilation_v(dilation.begin(), dilation.end());
    // mkldnn_reorder_conv2d_weight is dispatched on the MkldnnCPU backend,
    // so we first convert the NCHW CPU tensor into a dense MKLDNN tensor,
    // then let oneDNN reorder it into its preferred blocked format.
    auto mkldnn_weight = weight->to_mkldnn();
    auto result = at::mkldnn_reorder_conv2d_weight(
        mkldnn_weight,
        at::IntArrayRef(padding_v),
        at::IntArrayRef(stride_v),
        at::IntArrayRef(dilation_v),
        groups,
        c10::nullopt);
    return std::make_shared<CrossTensor>(std::move(result));
}

std::shared_ptr<CrossTensor> aten_mkldnn_convolution(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &packed_weight,
    TensorList bias_opt,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> dilation,
    int64_t groups)
{
    std::vector<int64_t> padding_v(padding.begin(), padding.end());
    std::vector<int64_t> stride_v(stride.begin(), stride.end());
    std::vector<int64_t> dilation_v(dilation.begin(), dilation.end());
    auto bias = opt_tensor(bias_opt);
    auto result = at::mkldnn_convolution(
        *input,
        *packed_weight,
        bias,
        at::IntArrayRef(padding_v),
        at::IntArrayRef(stride_v),
        at::IntArrayRef(dilation_v),
        groups);
    return std::make_shared<CrossTensor>(std::move(result));
}
