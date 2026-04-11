defmodule ExTorch.Export.CompiledGraph do
  @moduledoc """
  A pre-compiled graph executor for zero-overhead inference.

  Created by `ExTorch.Export.compile/2` at load time. Holds pre-resolved
  C++ operator handles and integer-indexed argument templates, eliminating
  all per-op overhead during `forward_compiled/2`.
  """

  @type t :: %__MODULE__{
          resource: reference(),
          reference: reference()
        }

  defstruct [:resource, :reference]
end
