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

// ============================================================================
// Direct ATen functional ops (Phase A)
//
// These bypass the torch::nn::Module wrapping and call libtorch's at::
// functional API directly. They are stateless: parameters are passed in by
// value rather than copied into a layer object every call. Used by
// ExTorch.Export's interpreter for high-overhead ops where the per-call
// Module construction was the bottleneck.
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
    bool batch_first);

TensorList aten_gru(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first);

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
    const std::shared_ptr<CrossTensor> &ffn_bias_2);

std::shared_ptr<CrossTensor> aten_scaled_dot_product_attention(
    const std::shared_ptr<CrossTensor> &query,
    const std::shared_ptr<CrossTensor> &key,
    const std::shared_ptr<CrossTensor> &value,
    double dropout_p,
    bool is_causal,
    double scale,
    bool has_scale);

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
    bool average_attn_weights);

// ============================================================================
// Direct ATen functional ops (Phase B): conv, norm, pool
//
// These replace the per-call torch::nn::Module construction in the
// interpreter. Optional tensors are passed via a TensorList of 0 or 1
// elements (size 0 = nullopt). Spatial args (stride/padding/dilation/etc.)
// are passed as Vec<i64> for per-axis support.
// ============================================================================

std::shared_ptr<CrossTensor> aten_conv2d(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    int64_t groups);

std::shared_ptr<CrossTensor> aten_conv1d(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    int64_t groups);

std::shared_ptr<CrossTensor> aten_conv3d(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    int64_t groups);

std::shared_ptr<CrossTensor> aten_convolution(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    bool transposed,
    rust::Vec<int64_t> output_padding,
    int64_t groups);

std::shared_ptr<CrossTensor> aten_conv_transpose2d(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &weight,
    TensorList bias_opt,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> output_padding,
    int64_t groups,
    rust::Vec<int64_t> dilation);

std::shared_ptr<CrossTensor> aten_batch_norm(
    const std::shared_ptr<CrossTensor> &input,
    TensorList weight_opt,
    TensorList bias_opt,
    TensorList running_mean_opt,
    TensorList running_var_opt,
    bool training,
    double momentum,
    double eps);

std::shared_ptr<CrossTensor> aten_layer_norm(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> normalized_shape,
    TensorList weight_opt,
    TensorList bias_opt,
    double eps);

std::shared_ptr<CrossTensor> aten_group_norm(
    const std::shared_ptr<CrossTensor> &input,
    int64_t num_groups,
    TensorList weight_opt,
    TensorList bias_opt,
    double eps);

std::shared_ptr<CrossTensor> aten_max_pool2d(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> kernel_size,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> dilation,
    bool ceil_mode);

std::shared_ptr<CrossTensor> aten_avg_pool2d(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> kernel_size,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> padding,
    bool ceil_mode,
    bool count_include_pad);

std::shared_ptr<CrossTensor> aten_adaptive_avg_pool2d(
    const std::shared_ptr<CrossTensor> &input,
    rust::Vec<int64_t> output_size);

// Thread pool inspection / control
int64_t aten_get_num_threads();
void aten_set_num_threads(int64_t n);
int64_t aten_get_num_interop_threads();
void aten_set_num_interop_threads(int64_t n);
bool aten_mkldnn_is_available();

// Identity (used to measure pure NIF marshaling overhead vs op kernel time)
std::shared_ptr<CrossTensor> aten_noop(const std::shared_ptr<CrossTensor> &t);

// Backend / context introspection
rust::String aten_backend_info();

// Global grad-enabled flag (PyTorch calls this torch::GradMode).
// When false, autograd skips tracking entirely and MKLDNN dispatch can
// pick inference-mode primitives. Setting this process-wide avoids
// per-call NoGradGuard overhead in the hot op path.
bool aten_is_grad_enabled();
void aten_set_grad_enabled(bool enabled);

// Clear the current OS thread's CPU affinity mask so OpenMP workers
// spawned by libtorch can use all cores. BEAM scheduler threads are
// typically bound to specific cores, and nested OpenMP workers inherit
// that mask — causing oversubscription on one core.
bool aten_clear_cpu_affinity();

// Block until all queued CUDA kernels have completed on the current
// stream (and optionally all streams on the current device). Required
// before `:timer.tc` measurements of GPU work, since CUDA launches are
// asynchronous and return before the kernel runs.
void aten_cuda_synchronize();

// MKLDNN weight pre-packing: reorder an NCHW conv weight into the
// MKLDNN blocked format that oneDNN prefers, once at load time, so
// the per-call reorder cost disappears from every forward pass.
std::shared_ptr<CrossTensor> aten_mkldnn_reorder_conv2d_weight(
    const std::shared_ptr<CrossTensor> &weight,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> dilation,
    int64_t groups);

// Run a convolution using an MKLDNN pre-packed weight. Skips the
// at::conv2d dispatch overhead and the per-call weight reorder.
std::shared_ptr<CrossTensor> aten_mkldnn_convolution(
    const std::shared_ptr<CrossTensor> &input,
    const std::shared_ptr<CrossTensor> &packed_weight,
    TensorList bias_opt,
    rust::Vec<int64_t> padding,
    rust::Vec<int64_t> stride,
    rust::Vec<int64_t> dilation,
    int64_t groups);
