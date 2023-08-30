defmodule ExTorch.Native.Tensor.Info do
  @moduledoc false

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
    Get the `requires_grad` status of a tensor.

    ## Arguments
      - tensor (`ExTorch.Tensor`): Input tensor
    """
    @spec requires_grad(ExTorch.Tensor.t()) :: boolean()
    defbinding(requires_grad(tensor))

    @doc """
    Get a human readable representation of a tensor.

    ## Arguments
      - tensor (`ExTorch.Tensor`): Input tensor

    ## Keyword args
    - `precision`: Number of digits of precision for floating point output. Default: 4

    - `threshold`: Total number of array elements which trigger summarization
    rather than full `repr`. Default: 1000.

    - `edgeitems`: Number of array items in summary at beginning and end of
    each dimension. Default: 3.

    - `linewidth`: The number of characters per line for the purpose of
      inserting line breaks (default = 80). Thresholded matrices will
      ignore this parameter.

    - `sci_mode`: Enable (`true`) or disable (`false`) scientific notation. If
      `nil` (default) is specified, the value is defined by
      the formatter. This value is automatically chosen by the framework.
    """
    @spec repr(ExTorch.Tensor.t(), ExTorch.Utils.PrintOptions.t()) :: binary()
    defbinding(repr(tensor, opts \\ %ExTorch.Utils.PrintOptions{}))

    @doc """
    Convert a tensor into a list.

    ## Arguments
      - tensor (`ExTorch.Tensor`): Input tensor
    """
    @spec to_list(ExTorch.Tensor.t()) :: list()
    defbinding(to_list(tensor))

    @doc """
    Returns the total number of elements in the input tensor.

    ## Arguments
      - `tensor` (`ExTorch.Tensor`): Input tensor.
    """
    @spec numel(ExTorch.Tensor.t()) :: integer()
    defbinding(numel(tensor))
  end
end
