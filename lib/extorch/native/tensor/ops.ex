defmodule ExTorch.Native.Tensor.Ops do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_manipulation) do
    @doc """
    Append an empty dimension to a tensor on a given dimension.

    ## Arguments
      - `tensor`: Input tensor (`ExTorch.Tensor`)
      - `dim`: Dimension (`integer()`)
    """
    @spec unsqueeze(
            ExTorch.Tensor.t(),
            integer()
          ) :: ExTorch.Tensor.t()
    defbinding(unsqueeze(tensor, dim))
  end
end
