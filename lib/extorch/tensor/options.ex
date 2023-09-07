defmodule ExTorch.Tensor.Options do
  @moduledoc """
  The ``ExTorch.Tensor.Options`` struct defines the creation parameters of a tensor.
  """

  @after_compile __MODULE__
  def __after_compile__(_env, bytecode) do
    __MODULE__
    |> :code.which()
    |> to_string()
    |> File.write(bytecode)
  end

  @typedoc """
  An ``ExTorch.Tensor.Options`` defines the creation parameters of a tensor.
  """
  @type t :: %__MODULE__{
          dtype: ExTorch.DType.dtype(),
          layout: ExTorch.Layout.layout(),
          device: ExTorch.Device.device(),
          requires_grad: boolean(),
          pin_memory: boolean(),
          memory_format: ExTorch.MemoryFormat.memory_format()
        }

  defstruct [
    # Type of the tensor reference
    dtype: nil,

    # The layout format of the tensor
    layout: :strided,

    # Device where the tensor lives on
    device: nil,

    # true if the tensor will accumulate gradients, false otherwise
    requires_grad: false,

    # true if the tensor will pin memory on a GPU device
    pin_memory: false,

    # The memory format which the tensor should have
    memory_format: :contiguous
  ]

  defimpl ExTorch.Protocol.DefaultStruct, for: __MODULE__ do
    @spec replace_defaults(ExTorch.Tensor.Options.t()) :: ExTorch.Tensor.Options.t()
    def replace_defaults(%ExTorch.Tensor.Options{dtype: dtype, device: device} = in_struct) do
      dtype =
        case dtype do
          nil -> ExTorch.get_default_dtype()
          _ -> dtype
        end

      device =
        case device do
          nil -> ExTorch.get_default_device()
          _ -> device
        end

      struct(in_struct, dtype: dtype, device: device)
    end
  end

  @doc false
  @spec merge_input(ExTorch.Tensor.t(), ExTorch.Tensor.Options.t()) :: ExTorch.Tensor.Options.t()
  def merge_input(%ExTorch.Tensor{} = input, options = %ExTorch.Tensor.Options{}) do
    dtype =
      case options.dtype do
        :auto -> ExTorch.Tensor.dtype(input)
        dtype -> dtype
      end

    device =
      case options.device do
        :auto -> ExTorch.Tensor.device(input)
        device -> device
      end

    layout =
      case options.layout do
        nil -> ExTorch.Tensor.layout(input)
        layout -> layout
      end

    mem_fmt =
      case options.memory_format do
        :preserve -> ExTorch.Tensor.memory_format(input)
        mem_fmt -> mem_fmt
      end

    struct(options, dtype: dtype, device: device, layout: layout, memory_format: mem_fmt)
  end
end
