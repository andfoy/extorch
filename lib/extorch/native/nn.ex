defmodule ExTorch.Native.NN do
  @moduledoc false

  defmacro __using__(_opts) do
    quote do
      # Layer factories
      @doc false
      def nn_linear(_in_features, _out_features, _bias), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_conv1d(_in_ch, _out_ch, _kernel, _stride, _pad, _dil, _groups, _bias), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_conv2d(_in_ch, _out_ch, _kernel, _stride, _pad, _dil, _groups, _bias), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_batch_norm1d(_features, _eps, _momentum, _affine, _track), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_batch_norm2d(_features, _eps, _momentum, _affine, _track), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_layer_norm(_shape, _eps, _affine), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_dropout(_p, _inplace), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_embedding(_num, _dim, _pad_idx, _has_pad), do: :erlang.nif_error(:nif_not_loaded)

      # Additional convolutions
      @doc false
      def nn_conv3d(_in, _out, _k, _s, _p, _d, _g, _b), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_conv_transpose1d(_in, _out, _k, _s, _p, _op, _g, _b, _d), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_conv_transpose2d(_in, _out, _k, _s, _p, _op, _g, _b, _d), do: :erlang.nif_error(:nif_not_loaded)

      # Pooling
      @doc false
      def nn_max_pool1d(_k, _s, _p, _d, _c), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_max_pool2d(_k, _s, _p, _d, _c), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_avg_pool1d(_k, _s, _p, _c, _ci), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_avg_pool2d(_k, _s, _p, _c, _ci), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_adaptive_avg_pool1d(_o), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_adaptive_avg_pool2d(_oh, _ow), do: :erlang.nif_error(:nif_not_loaded)

      # Additional normalization
      @doc false
      def nn_group_norm(_g, _c, _e, _a), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_instance_norm1d(_f, _e, _m, _a, _t), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_instance_norm2d(_f, _e, _m, _a, _t), do: :erlang.nif_error(:nif_not_loaded)

      # Recurrent
      @doc false
      def nn_lstm(_i, _h, _n, _b, _bf, _d, _bi), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_gru(_i, _h, _n, _b, _bf, _d, _bi), do: :erlang.nif_error(:nif_not_loaded)

      # Transformer
      @doc false
      def nn_multihead_attention(_e, _h, _d, _b), do: :erlang.nif_error(:nif_not_loaded)

      # Utility
      @doc false
      def nn_flatten(_s, _e), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_unflatten(_d, _s), do: :erlang.nif_error(:nif_not_loaded)

      # Activations
      @doc false
      def nn_relu(_inplace), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_gelu(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_sigmoid(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_tanh(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_softmax(_dim), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_leaky_relu(_slope, _inplace), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_elu(_alpha, _inplace), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_silu(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_mish(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_prelu(_n), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_log_softmax(_dim), do: :erlang.nif_error(:nif_not_loaded)

      # Module operations
      @doc false
      def nn_forward(_module, _input), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_parameters(_module), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_set_eval(_module), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_set_train(_module), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_type_name(_module), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_copy_parameters(_module, _params), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def nn_to_device(_module, _device), do: :erlang.nif_error(:nif_not_loaded)

      # Direct ATen functional ops (Phase A)
      @doc false
      def aten_lstm(_input, _hx, _params, _has_biases, _num_layers, _dropout, _train, _bidirectional, _batch_first),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_gru(_input, _hx, _params, _has_biases, _num_layers, _dropout, _train, _bidirectional, _batch_first),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_transformer_encoder_layer_fwd(
            _src, _embed_dim, _num_heads,
            _qkv_w, _qkv_b, _proj_w, _proj_b,
            _use_gelu, _norm_first, _eps,
            _n1_w, _n1_b, _n2_w, _n2_b,
            _ffn1_w, _ffn1_b, _ffn2_w, _ffn2_b
          ),
          do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_scaled_dot_product_attention(_q, _k, _v, _dropout_p, _is_causal, _scale, _has_scale),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_native_multi_head_attention(
            _q, _k, _v, _embed_dim, _num_head,
            _qkv_w, _qkv_b, _proj_w, _proj_b,
            _need_weights, _average_attn_weights
          ),
          do: :erlang.nif_error(:nif_not_loaded)

      # Direct ATen functional ops (Phase B): conv, norm, pool
      @doc false
      def aten_conv2d(_input, _weight, _bias_opt, _stride, _padding, _dilation, _groups),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_conv1d(_input, _weight, _bias_opt, _stride, _padding, _dilation, _groups),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_conv3d(_input, _weight, _bias_opt, _stride, _padding, _dilation, _groups),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_convolution(_input, _weight, _bias_opt, _stride, _padding, _dilation, _transposed, _output_padding, _groups),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_conv_transpose2d(_input, _weight, _bias_opt, _stride, _padding, _output_padding, _groups, _dilation),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_batch_norm(_input, _weight_opt, _bias_opt, _running_mean_opt, _running_var_opt, _training, _momentum, _eps),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_layer_norm(_input, _normalized_shape, _weight_opt, _bias_opt, _eps),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_group_norm(_input, _num_groups, _weight_opt, _bias_opt, _eps),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_max_pool2d(_input, _kernel_size, _stride, _padding, _dilation, _ceil_mode),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_avg_pool2d(_input, _kernel_size, _stride, _padding, _ceil_mode, _count_include_pad),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_adaptive_avg_pool2d(_input, _output_size),
        do: :erlang.nif_error(:nif_not_loaded)

      # Thread pool / backend inspection
      @doc false
      def aten_get_num_threads(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_set_num_threads(_n), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_get_num_interop_threads(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_set_num_interop_threads(_n), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_mkldnn_is_available(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_noop(_t), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_backend_info(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_is_grad_enabled(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_set_grad_enabled(_enabled), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_clear_cpu_affinity(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_cuda_synchronize(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_mkldnn_reorder_conv2d_weight(_w, _pad, _stride, _dil, _groups),
        do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aten_mkldnn_convolution(_input, _packed_w, _bias_opt, _pad, _stride, _dil, _groups),
        do: :erlang.nif_error(:nif_not_loaded)
    end
  end
end
