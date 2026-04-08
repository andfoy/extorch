defmodule ExTorch.JIT.Model do
  @moduledoc """
  Represents a loaded TorchScript model.

  A `%ExTorch.JIT.Model{}` struct wraps a reference to a `torch::jit::script::Module`
  loaded from a `.pt` file. The model can be used for inference via `ExTorch.JIT.forward/2`.
  """

  @type t :: %__MODULE__{
          resource: any(),
          reference: reference(),
          device: ExTorch.Device.device()
        }

  defstruct resource: nil,
            reference: nil,
            device: :cpu
end

defimpl Inspect, for: ExTorch.JIT.Model do
  def inspect(%ExTorch.JIT.Model{device: device}, _opts) do
    "#JIT.Model<[device: #{inspect(device)}]>"
  end
end
