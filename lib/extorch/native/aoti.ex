defmodule ExTorch.Native.AOTI do
  @moduledoc false

  defmacro __using__(_opts) do
    quote do
      @doc false
      def aoti_is_available(), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aoti_load(_path, _model_name, _device_index), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aoti_forward(_model, _inputs), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aoti_get_metadata_keys(_model), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aoti_get_metadata_value(_model, _key), do: :erlang.nif_error(:nif_not_loaded)
      @doc false
      def aoti_get_constant_fqns(_model), do: :erlang.nif_error(:nif_not_loaded)
    end
  end
end
