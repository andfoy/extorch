defmodule ExTorch.Tensor.BlobView do
  @moduledoc """
  A tensor view backed by foreign memory.

  This struct holds both the `%ExTorch.Tensor{}` and a reference to the memory
  owner, ensuring the owner is not garbage collected while the view is alive.

  Access the underlying tensor via the `tensor` field. The `owner` field keeps
  the source data alive.
  """

  @type t :: %__MODULE__{
          tensor: ExTorch.Tensor.t(),
          owner: term()
        }

  defstruct [:tensor, :owner]
end

defimpl Inspect, for: ExTorch.Tensor.BlobView do
  def inspect(%ExTorch.Tensor.BlobView{tensor: tensor}, opts) do
    "#BlobView<#{Inspect.ExTorch.Tensor.inspect(tensor, opts)}>"
  end
end
