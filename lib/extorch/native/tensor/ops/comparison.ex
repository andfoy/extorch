defmodule ExTorch.Native.Tensor.Ops.Comparison do
  @moduledoc false
  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_comparison) do
    @doc ~S"""
    This function checks if `input` and `other` satisfy the condition:
    $$|\text{input} - \text{other}| \leq \texttt{atol} + \texttt{rtol} \times |\text{other}|$$
    elementwise, for all elements of `input` and `other`.

    ## Arguments
    - `input` - First tensor to compare (`ExTorch.Tensor`)
    - `other` - Second tensor to compare (`ExTorch.Tensor`)

    ## Optional arguments
    - `rtol` - Relative tolerance (`float`). Default: 1.0e-5
    - `atol` - Absolute tolerance (`float`). Default: 1.0e-8
    - `equal_nan` - If `true`, then two `NaN`s will be considered equal. Default: `false`.

    ## Examples
        iex> ExTorch.allclose(ExTorch.tensor([10000.0, 1.0e-07]), ExTorch.tensor([10000.1, 1.0e-08]))
        false
        iex> ExTorch.allclose(ExTorch.tensor([10000.0, 1.0e-08]), ExTorch.tensor([10000.1, 1.0e-09]))
        true
        iex> ExTorch.allclose(ExTorch.tensor([1.0, :nan]), ExTorch.tensor([1.0, :nan]))
        false
        iex> ExTorch.allclose(ExTorch.tensor([1.0, :nan]), ExTorch.tensor([1.0, :nan]), equal_nan: true)
        true
    """
    @spec allclose(ExTorch.Tensor.t(), ExTorch.Tensor.t(), float(), float(), boolean()) ::
            boolean()
    defbinding(allclose(input, other, rtol \\ 1.0e-5, atol \\ 1.0e-8, equal_nan \\ false))
  end
end
