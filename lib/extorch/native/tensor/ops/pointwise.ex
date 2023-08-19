defmodule ExTorch.Native.Tensor.Ops.PointWise do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_pointwise) do
    @doc """
    Returns a new tensor containing real values of the `input` tensor.
    The returned tensor and `input` share the same underlying storage.

    ## Arguments
    - `input`: The input tensor.
    """
    @spec real(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(real(input))

    @doc """
    Returns a new tensor containing imaginary values of the `input` tensor.
    The returned tensor and `input` share the same underlying storage.

    ## Arguments
    - `input`: The input tensor.
    """
    @spec imag(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(imag(input))
  end
end
