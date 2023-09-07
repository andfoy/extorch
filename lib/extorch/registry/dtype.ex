defmodule ExTorch.Registry.DType do
  @moduledoc false

  @after_compile __MODULE__
  def __after_compile__(_env, bytecode) do
    __MODULE__
    |> :code.which()
    |> to_string()
    |> File.write(bytecode)
  end

  require ExTorch.DType

  @doc """
  Get the current default floating point dtype for the current process.

  ## Notes
  By default, `ExTorch` will set the default dtype to `:float32`.
  """
  @doc kind: :process_values
  @spec get_default_dtype() :: ExTorch.DType.dtype()
  def get_default_dtype() do
    case Registry.select(__MODULE__, [{{:dtype, self(), :"$1"}, [], [:"$1"]}]) do
      [] ->
        Registry.register(__MODULE__, :dtype, :float32)
        :float32
      [value] -> value
    end
  end

  @doc """
  Sets the default floating point dtype of the current process to `dtype`.

  Supports `:float32` and `:float64` as inputs. Other dtypes may be accepted
  without complaint but are not supported and are unlikely to work as expected.

  When PyTorch is initialized its default floating point dtype is `:float32`,
  and the intent of `set_default_dtype(:float64)` is to facilitate NumPy-like
  type inference. The default floating point dtype is used to:

  1. Implicitly determine the default complex dtype. When the default floating
     point type is `:float32` the default complex dtype is `:complex64`, and
     when the default floating point type is `:float64` the default complex
     type is `:complex128`.

  2. Infer the dtype for tensors constructed using Elixir floats or
     `ExTorch.Complex` numbers. See examples below.

  3. Determine the result of type promotion between bool and integer tensors and
     Elixir floats and `ExTorch.Complex` numbers.
  """
  @doc kind: :process_values
  @spec set_default_dtype(ExTorch.DType.dtype()) :: ExTorch.DType.dtype()
  def set_default_dtype(dtype) when ExTorch.DType.is_dtype(dtype) do
    case Registry.values(__MODULE__, :dtype, self()) do
      [] ->
        Registry.register(__MODULE__, :dtype, dtype)
      _ ->
        Registry.unregister(__MODULE__, :dtype)
        Registry.register(__MODULE__, :dtype, dtype)
    end

    dtype
  end
end
