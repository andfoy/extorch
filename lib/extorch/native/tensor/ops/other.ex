defmodule ExTorch.Native.Tensor.Ops.Other do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_other_ops) do
    @doc ~S"""
    Returns a view of `input` as a complex tensor.

    For an input complex tensor of size $$(\text{m1}, \text{m2}, \cdots, \text{mi}, 2)$$,
    this function returns a new complex tensor of size $$(\text{m1}, \text{m2}, \cdots, \text{mi})$$
    where the last dimension of the input tensor is expected to represent the real
    and imaginary components of complex numbers.

    ## Arguments
    - input: The input `ExTorch.Tensor`

    ## Notes
    `view_as_complex/1` is only supported for tensors with `ExTorch.DType`
    `:float64` and `:float32`. The input is expected to have the last dimension
    of size 2. In addition, the tensor must have a stride of 1 for its last
    dimension. The strides of all other dimensions must be even numbers.

    ## Examples
        iex> x = ExTorch.randint(-3, 3, {5, 2})
        #Tensor<
        [[ 2., -1.],
         [ 0., -1.],
         [-2., -2.],
         [ 2.,  0.],
         [ 1., -1.]]
        [
          size: {5, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.view_as_complex(x)
        #Tensor<
        [ 2.-1.j,  0.-1.j, -2.-2.j,  2.+0.j,  1.-1.j]
        [
          size: {5},
          dtype: :complex_double,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec view_as_complex(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(view_as_complex(input))
  end
end
