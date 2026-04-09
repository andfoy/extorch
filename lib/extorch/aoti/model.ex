defmodule ExTorch.AOTI.Model do
  @moduledoc """
  Represents a loaded AOTI (.pt2) compiled model.

  A `%ExTorch.AOTI.Model{}` wraps a reference to a
  `torch::inductor::AOTIModelPackageLoader` loaded from a `.pt2` package.
  """

  @type t :: %__MODULE__{
          resource: any(),
          reference: reference()
        }

  defstruct resource: nil,
            reference: nil
end

defimpl Inspect, for: ExTorch.AOTI.Model do
  def inspect(%ExTorch.AOTI.Model{}, _opts) do
    "#AOTI.Model<>"
  end
end
