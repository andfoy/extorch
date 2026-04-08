#pragma once
#include "common.h"
#include "utils.h"
#include <torch/torch.h>

// CrossNNModule wraps a torch::nn::AnyModule which can hold any nn::Module.
// This allows us to store heterogeneous module types behind a single pointer.
struct CrossNNModuleImpl {
    torch::nn::AnyModule module;
    std::string type_name;

    CrossNNModuleImpl(torch::nn::AnyModule m, std::string name)
        : module(std::move(m)), type_name(std::move(name)) {}
};

struct CrossNNModuleRef;
struct NNModuleParam;

// Layer factory functions
std::shared_ptr<CrossNNModule> nn_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias);

std::shared_ptr<CrossNNModule> nn_conv1d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool bias);

std::shared_ptr<CrossNNModule> nn_conv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool bias);

std::shared_ptr<CrossNNModule> nn_batch_norm1d(
    int64_t num_features,
    double eps,
    double momentum,
    bool affine,
    bool track_running_stats);

std::shared_ptr<CrossNNModule> nn_batch_norm2d(
    int64_t num_features,
    double eps,
    double momentum,
    bool affine,
    bool track_running_stats);

std::shared_ptr<CrossNNModule> nn_layer_norm(
    rust::Vec<int64_t> normalized_shape,
    double eps,
    bool elementwise_affine);

std::shared_ptr<CrossNNModule> nn_dropout(double p, bool inplace);

std::shared_ptr<CrossNNModule> nn_embedding(
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t padding_idx,
    bool has_padding_idx);

// Additional convolutions
std::shared_ptr<CrossNNModule> nn_conv3d(
    int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool bias);

std::shared_ptr<CrossNNModule> nn_conv_transpose1d(
    int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t output_padding, int64_t groups, bool bias, int64_t dilation);

std::shared_ptr<CrossNNModule> nn_conv_transpose2d(
    int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t output_padding, int64_t groups, bool bias, int64_t dilation);

// Pooling
std::shared_ptr<CrossNNModule> nn_max_pool1d(int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode);
std::shared_ptr<CrossNNModule> nn_max_pool2d(int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode);
std::shared_ptr<CrossNNModule> nn_avg_pool1d(int64_t kernel_size, int64_t stride, int64_t padding, bool ceil_mode, bool count_include_pad);
std::shared_ptr<CrossNNModule> nn_avg_pool2d(int64_t kernel_size, int64_t stride, int64_t padding, bool ceil_mode, bool count_include_pad);
std::shared_ptr<CrossNNModule> nn_adaptive_avg_pool1d(int64_t output_size);
std::shared_ptr<CrossNNModule> nn_adaptive_avg_pool2d(int64_t output_h, int64_t output_w);

// Additional normalization
std::shared_ptr<CrossNNModule> nn_group_norm(int64_t num_groups, int64_t num_channels, double eps, bool affine);
std::shared_ptr<CrossNNModule> nn_instance_norm1d(int64_t num_features, double eps, double momentum, bool affine, bool track_running_stats);
std::shared_ptr<CrossNNModule> nn_instance_norm2d(int64_t num_features, double eps, double momentum, bool affine, bool track_running_stats);

// Recurrent
std::shared_ptr<CrossNNModule> nn_lstm(int64_t input_size, int64_t hidden_size, int64_t num_layers, bool bias, bool batch_first, double dropout, bool bidirectional);
std::shared_ptr<CrossNNModule> nn_gru(int64_t input_size, int64_t hidden_size, int64_t num_layers, bool bias, bool batch_first, double dropout, bool bidirectional);

// Transformer
std::shared_ptr<CrossNNModule> nn_multihead_attention(int64_t embed_dim, int64_t num_heads, double dropout, bool bias);

// Utility
std::shared_ptr<CrossNNModule> nn_flatten(int64_t start_dim, int64_t end_dim);
std::shared_ptr<CrossNNModule> nn_unflatten(int64_t dim, rust::Vec<int64_t> sizes);

// Activation functions
std::shared_ptr<CrossNNModule> nn_relu(bool inplace);
std::shared_ptr<CrossNNModule> nn_gelu();
std::shared_ptr<CrossNNModule> nn_sigmoid();
std::shared_ptr<CrossNNModule> nn_tanh();
std::shared_ptr<CrossNNModule> nn_softmax(int64_t dim);
std::shared_ptr<CrossNNModule> nn_leaky_relu(double negative_slope, bool inplace);
std::shared_ptr<CrossNNModule> nn_elu(double alpha, bool inplace);
std::shared_ptr<CrossNNModule> nn_silu();
std::shared_ptr<CrossNNModule> nn_mish();
std::shared_ptr<CrossNNModule> nn_prelu(int64_t num_parameters);
std::shared_ptr<CrossNNModule> nn_log_softmax(int64_t dim);

// Module operations
std::shared_ptr<CrossTensor> nn_forward(
    const std::shared_ptr<CrossNNModule> &module,
    const std::shared_ptr<CrossTensor> &input);

rust::Vec<NamedTensor> nn_parameters(
    const std::shared_ptr<CrossNNModule> &module);

void nn_set_eval(const std::shared_ptr<CrossNNModule> &module);
void nn_set_train(const std::shared_ptr<CrossNNModule> &module);

rust::String nn_type_name(const std::shared_ptr<CrossNNModule> &module);

void nn_copy_parameters(
    const std::shared_ptr<CrossNNModule> &dst,
    rust::Vec<NamedTensor> params);

std::shared_ptr<CrossNNModule> nn_to_device(
    const std::shared_ptr<CrossNNModule> &module,
    Device s_device);
