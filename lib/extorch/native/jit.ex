defmodule ExTorch.Native.JIT do
  @moduledoc false

  defmacro __using__(_opts) do
    quote do
      @doc false
      def jit_load(_path, _device), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_save(_model, _path), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_forward(_model, _inputs), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_invoke_method(_model, _method_name, _inputs), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_get_method_names(_model), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_named_parameters(_model), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_named_buffers(_model), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_named_modules(_model), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_set_eval(_model), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_set_train(_model), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def jit_to_device(_model, _device), do: :erlang.nif_error(:nif_not_loaded)

      # IR Introspection
      @doc false
      def jit_graph_str(_model), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def jit_module_parameters_info(_model), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def jit_module_submodules_info(_model), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def jit_module_methods_info(_model), do: :erlang.nif_error(:nif_not_loaded)
    end
  end
end
