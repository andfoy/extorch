defmodule ExTorch.Native.Tensor.Ops.Mutating do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_mutating) do
    @doc """
    Returns a view of `input` with a flipped conjugate bit.
    If `input` has a non-complex dtype, this function just returns `input`.

    ## Arguments
      - `tensor`: Input tensor (`ExTorch.Tensor`)

    ## Notes
    `ExTorch.conj/1` performs a lazy conjugation, but the actual conjugated
    tensor can be materialized at any time using `ExTorch.resolve_conj/1`.

    ## Examples
        iex> a = ExTorch.rand({2, 2}, dtype: :complex64)
        #Tensor<
        [[0.5885+0.0263j, 0.8141+0.0605j],
         [0.9169+0.3126j, 0.6344+0.2768j]]
        [
          size: {2, 2},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

        # Conjugate the input
        iex> b = ExTorch.conj(a)
        #Tensor<
        [[0.5885-0.0263j, 0.8141-0.0605j],
         [0.9169-0.3126j, 0.6344-0.2768j]]
        [
          size: {2, 2},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

        # Check that conj bit is set to true
        iex> ExTorch.Tensor.is_conj(b)
        true

    """
    @spec conj(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(conj(tensor))
  end
end
