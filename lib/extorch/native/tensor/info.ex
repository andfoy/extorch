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
    Get the `memory_format` of a tensor.

    ## Arguments
      - tensor (`ExTorch.Tensor`): Input tensor
    """
    @spec memory_format(ExTorch.Tensor.t()) :: ExTorch.MemoryFormat.memory_format()
    defbinding(memory_format(tensor))

    @doc """
    Get the `layout` of a tensor.

    ## Arguments
      - tensor (`ExTorch.Tensor`): Input tensor
    """
    @spec layout(ExTorch.Tensor.t()) :: ExTorch.Layout.layout()
    defbinding(layout(tensor))

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

    ## Examples
        iex> x = ExTorch.empty({3, 4, 5})
        iex> ExTorch.Tensor.numel(x)
        60
    """
    @spec numel(ExTorch.Tensor.t()) :: integer()
    defbinding(numel(tensor))

    @doc """
    Returns `true` if the datatype of `input` is a complex data type. i.e., one
    of `:complex64` or `:complex128`

    ## Arguments
      - `tensor` (`ExTorch.Tensor`): Input tensor.

    ## Examples
        # Complex tensors yield true
        iex> a = ExTorch.rand({2, 2}, dtype: :complex64)
        iex> ExTorch.Tensor.is_complex(a)
        true

        # Non-complex tensors yield false
        iex> b = ExTorch.randint(3, {2, 2}, dtype: :int32)
        iex> ExTorch.Tensor.is_complex(b)
        false

    """
    @spec is_complex(ExTorch.Tensor.t()) :: boolean()
    defbinding(is_complex(tensor))

    @doc """
    Returns `true` if the datatype of `input` is a floating data type. i.e., one
    of `:float16`, `:float32`, `:float64` or `:bfloat16`.

    ## Arguments
      - `tensor` (`ExTorch.Tensor`): Input tensor.

    ## Examples
        # Floating-type tensors yield true
        iex> a = ExTorch.rand({2, 2}, dtype: :float16)
        iex> ExTorch.Tensor.is_floating_point(a)
        true

        # Other type of tensors yield false
        iex> b = ExTorch.rand({2, 2}, dtype: :complex128)
        iex> ExTorch.Tensor.is_floating_point(b)
        false
    """
    @spec is_floating_point(ExTorch.Tensor.t()) :: boolean()
    defbinding(is_floating_point(tensor))

    @doc """
    Returns `true` if `input` is a conjugated tensor. i.e., its conjugate bit is
    set to `true`.

    ## Arguments
      - `tensor` (`ExTorch.Tensor`): Input tensor.

    ## Examples
        # Complex tensors have the conj bit set to false by default
        iex> a = ExTorch.rand({3, 4}, dtype: :complex64)
        iex> ExTorch.Tensor.is_conj(a)
        false

        # Conjugated tensor views have the conj bit set to true
        iex> b = ExTorch.conj(a)
        iex> ExTorch.Tensor.is_conj(b)
        true

        # Materialized conjugate tensors have the conj bit set to false
        iex> c = ExTorch.resolve_conj(b)
        iex> ExTorch.Tensor.is_conj(c)
        false
    """
    @spec is_conj(ExTorch.Tensor.t()) :: boolean()
    defbinding(is_conj(tensor))

    @doc """
    Returns `true` if the input is a single element tensor which is not equal to
    zero after type conversions.

    ## Arguments
      - `tensor` (`ExTorch.Tensor`): Input tensor.

    ## Examples
        iex> ExTorch.Tensor.is_nonzero(ExTorch.tensor([0.0]))
        false
        iex> ExTorch.Tensor.is_nonzero(ExTorch.tensor([1.5]))
        true
        iex> ExTorch.Tensor.is_nonzero(ExTorch.tensor([false]))
        false
        iex> ExTorch.Tensor.is_nonzero(ExTorch.tensor([3]))
        true
        iex> ExTorch.Tensor.is_nonzero(ExTorch.tensor([1, 2, 3]))
        ** (ErlangError) Erlang error: "Boolean value of Tensor with more than one value is ambiguous"
            (extorch 0.1.0-pre0) ExTorch.Native.is_nonzero(#Tensor<
        [1, 2, 3]
        [size: {3}, dtype: :byte, device: :cpu, requires_grad: false]>)
    """
    @spec is_nonzero(ExTorch.Tensor.t()) :: boolean()
    defbinding(is_nonzero(tensor))
  end
end
