defmodule ExTorch.Native.Dispatcher do
  @moduledoc false

  defmacro __using__(_opts) do
    quote do
      @doc false
      def load_torch_library(_path), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def dispatch_op(_op_name, _overload, _args), do: :erlang.nif_error(:nif_not_loaded)

      @doc false
      def list_registered_ops(_ns_prefix), do: :erlang.nif_error(:nif_not_loaded)
    end
  end
end
