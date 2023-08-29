defmodule ExTorch.Native.Tensor.Ops.PointWise do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_pointwise) do
    @doc """
    Returns a new tensor containing real values of the `input` tensor.
    The returned tensor and `input` share the same underlying storage.

    ## Arguments
    - `input`: The input tensor.

    ## Examples
        iex> x = ExTorch.rand({3}, dtype: :complex64)
        #Tensor<
        [0.8235+0.9395j, 0.9912+0.4506j, 0.5164+0.3070j]
        [
          size: {3},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.real(x)
        #Tensor<
        [0.8235, 0.9912, 0.5164]
        [
          size: {3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec real(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(real(input))

    @doc """
    Returns a new tensor containing imaginary values of the `input` tensor.
    The returned tensor and `input` share the same underlying storage.

    ## Arguments
    - `input`: The input tensor.

    ## Examples
        iex> x = ExTorch.rand({3}, dtype: :complex64)
        #Tensor<
        [0.8235+0.9395j, 0.9912+0.4506j, 0.5164+0.3070j]
        [
          size: {3},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.imag(x)
        #Tensor<
        [0.9395, 0.4506, 0.3070]
        [
          size: {3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec imag(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(imag(input))
  end
end
