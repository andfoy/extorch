#include "extorch/src/native.rs.h"
#include "extorch/include/nn.h"


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
