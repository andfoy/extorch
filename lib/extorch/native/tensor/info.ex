defmodule ExTorch.Native.Tensor.Info do
  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_info) do
    @doc """
    Get the size of a tensor.

    ## Arguments
      - `tensor`: Input tensor
    """
    @spec size(ExTorch.Tensor.t()) :: tuple()
    defbinding(size(tensor))

    @doc """
    Get the dtype of a tensor.

    ## Arguments
      - tensor (`ExTorch.Tensor`): Input tensor
    """
    @spec dtype(ExTorch.Tensor.t()) :: ExTorch.DType.dtype()
    defbinding(dtype(tensor))

    @doc """
    Get the device of a tensor.

    ## Arguments
      - tensor (`ExTorch.Tensor`): Input tensor
    """
    @spec device(ExTorch.Tensor.t()) :: ExTorch.Device.device()
    defbinding(device(tensor))

    @doc """
    Get a human readable representation of a tensor.

    ## Arguments
      - tensor (`ExTorch.Tensor`): Input tensor
    """
    @spec repr(ExTorch.Tensor.t()) :: binary()
    defbinding(repr(tensor))
  end
end
