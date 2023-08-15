defmodule ExTorch.Index.Slice do
  @moduledoc """
  Slice index definition.
  """

  @typedoc """
  An ``ExTorch.Index.Slice`` is a struct that represents an indexing slice
  with a start, stop and a number of steps between values.
  """
  @type t :: %__MODULE__{
          start: integer(),
          stop: integer(),
          step: integer(),
          mask: integer()
        }

  defstruct start: 0,
            stop: 0,
            step: 0,
            mask: 0
end
