defmodule ExTorch.NN.Layer do
  @moduledoc """
  Represents an instantiated neural network layer (nn.Module).

  A `%ExTorch.NN.Layer{}` wraps a reference to a `torch::nn::AnyModule`,
  which can hold any PyTorch layer type (Linear, Conv2d, ReLU, etc.).

  Use `ExTorch.NN` functions to create layers and run forward passes.
  """

  @type t :: %__MODULE__{
          resource: any(),
          reference: reference(),
          type_name: String.t()
        }

  defstruct resource: nil,
            reference: nil,
            type_name: ""
end

defimpl Inspect, for: ExTorch.NN.Layer do
  def inspect(%ExTorch.NN.Layer{type_name: type_name}, _opts) do
    "#NN.Layer<#{type_name}>"
  end
end
