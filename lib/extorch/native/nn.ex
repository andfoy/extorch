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
    end
  end
end
