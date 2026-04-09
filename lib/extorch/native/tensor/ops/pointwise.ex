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

    @doc """
    Adds `other` to `input`, scaled by `alpha`: $out = input + alpha \\times other$.

    ## Args
      * `input` (`ExTorch.Tensor`) - the first input tensor.
      * `other` (`ExTorch.Tensor`) - the second input tensor.
      * `alpha` (`number`) - the multiplier for `other`. Default: `1`.

    ## Shape
      * Input: `{*}` (any shape, must be broadcastable).
      * Output: same shape as broadcasted input.
    """
    @spec add(ExTorch.Tensor.t(), ExTorch.Tensor.t(), number()) :: ExTorch.Tensor.t()
    defbinding(add(input, other, alpha \\ 1))

    @doc """
    Subtracts `other` from `input`, scaled by `alpha`: $out = input - alpha \\times other$.

    ## Args
      * `input` (`ExTorch.Tensor`) - the first input tensor.
      * `other` (`ExTorch.Tensor`) - the second input tensor.
      * `alpha` (`number`) - the multiplier for `other`. Default: `1`.
    """
    @spec sub(ExTorch.Tensor.t(), ExTorch.Tensor.t(), number()) :: ExTorch.Tensor.t()
    defbinding(sub(input, other, alpha \\ 1))

    @doc """
    Multiplies `input` by `other` element-wise: $out_i = input_i \\times other_i$.

    ## Args
      * `input` (`ExTorch.Tensor`) - the first input tensor.
      * `other` (`ExTorch.Tensor`) - the second input tensor.
    """
    @spec mul(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(mul(input, other))

    @doc """
    Divides `input` by `other` element-wise: $out_i = \\frac{input_i}{other_i}$.

    ## Args
      * `input` (`ExTorch.Tensor`) - the dividend tensor.
      * `other` (`ExTorch.Tensor`) - the divisor tensor.
    """
    @spec tensor_div(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(tensor_div(input, other))

    @doc """
    Returns the negative of `input` element-wise: $out = -input$.
    """
    @spec neg(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(neg(input))

    @doc """
    Computes the absolute value of each element: $out_i = |input_i|$.
    """
    @spec tensor_abs(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(tensor_abs(input))

    @doc """
    Takes the power of each element by `exponent`: $out_i = input_i^{exponent}$.

    ## Args
      * `input` (`ExTorch.Tensor`) - the input tensor.
      * `exponent` (`number`) - the exponent value.
    """
    @spec pow_tensor(ExTorch.Tensor.t(), number()) :: ExTorch.Tensor.t()
    defbinding(pow_tensor(input, exponent))

    @doc """
    Returns a new tensor with the exponential: $out_i = e^{input_i}$.
    """
    @spec tensor_exp(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(tensor_exp(input))

    @doc """
    Returns a new tensor with the natural logarithm: $out_i = \\ln(input_i)$.
    """
    @spec tensor_log(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(tensor_log(input))

    @doc """
    Returns a new tensor with the square root: $out_i = \\sqrt{input_i}$.
    """
    @spec tensor_sqrt(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(tensor_sqrt(input))

    @doc """
    Returns a new tensor with the sine: $out_i = \\sin(input_i)$.
    """
    @spec tensor_sin(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(tensor_sin(input))

    @doc """
    Returns a new tensor with the cosine: $out_i = \\cos(input_i)$.
    """
    @spec tensor_cos(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(tensor_cos(input))

    @doc """
    Clamps all elements in `input` into the range `[min, max]`.

    $out_i = \\min(\\max(input_i, min), max)$

    ## Args
      * `input` (`ExTorch.Tensor`) - the input tensor.
      * `min` (`number`) - lower bound of the range.
      * `max` (`number`) - upper bound of the range.
    """
    @spec clamp(ExTorch.Tensor.t(), number(), number()) :: ExTorch.Tensor.t()
    defbinding(clamp(input, min_val, max_val))

    @doc """
    Matrix product of two tensors.

    The behavior depends on the dimensionality of the tensors:
      * If both are 1-D, computes the dot product.
      * If both are 2-D, computes matrix-matrix product.
      * If the first is 1-D and second is 2-D, a 1 is prepended to its dimension for
        the matrix multiply, then removed after.
      * If the first is 2-D and second is 1-D, computes matrix-vector product.
      * If both are at least 1-D and at least one is N-D (N > 2), computes a batched
        matrix multiply.

    ## Args
      * `input` (`ExTorch.Tensor`) - the first tensor.
      * `other` (`ExTorch.Tensor`) - the second tensor.
    """
    @spec matmul(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(matmul(input, other))

    @doc """
    Performs a matrix multiplication of the matrices `input` and `other`.

    If `input` is a `{n, m}` tensor, `other` is a `{m, p}` tensor,
    output will be a `{n, p}` tensor. For batched matrix multiply, see `bmm/2`.

    ## Args
      * `input` (`ExTorch.Tensor`) - the first matrix `{n, m}`.
      * `other` (`ExTorch.Tensor`) - the second matrix `{m, p}`.

    ## Shape
      * Input: `{n, m}` and `{m, p}`.
      * Output: `{n, p}`.
    """
    @spec mm(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(mm(input, other))

    @doc """
    Performs a batch matrix-matrix product of matrices stored in `input` and `other`.

    `input` and `other` must be 3-D tensors each containing the same number of matrices.

    ## Args
      * `input` (`ExTorch.Tensor`) - the first batch of matrices `{b, n, m}`.
      * `other` (`ExTorch.Tensor`) - the second batch of matrices `{b, m, p}`.

    ## Shape
      * Input: `{b, n, m}` and `{b, m, p}`.
      * Output: `{b, n, p}`.
    """
    @spec bmm(ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(bmm(input, other))

    @doc """
    Returns a tensor of elements selected from either `x` or `y`,
    depending on `condition`.

    $out_i = \\begin{cases} x_i & \\text{if } condition_i \\\\ y_i & \\text{otherwise} \\end{cases}$

    ## Args
      * `condition` (`ExTorch.Tensor`) - a boolean tensor.
      * `x` (`ExTorch.Tensor`) - values selected where condition is `true`.
      * `y` (`ExTorch.Tensor`) - values selected where condition is `false`.
    """
    @spec tensor_where(ExTorch.Tensor.t(), ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(tensor_where(condition, x, y))

    @doc """
    Fills elements of `input` tensor with `value` where `mask` is `true`.

    ## Args
      * `input` (`ExTorch.Tensor`) - the input tensor.
      * `mask` (`ExTorch.Tensor`) - a boolean mask tensor.
      * `value` (`number`) - the value to fill in where mask is true.
    """
    @spec masked_fill(ExTorch.Tensor.t(), ExTorch.Tensor.t(), number()) :: ExTorch.Tensor.t()
    defbinding(masked_fill(input, mask, value))

    @doc """
    Returns a contiguous in memory tensor containing the same data.
    If the tensor is already contiguous, returns itself.
    """
    @spec contiguous(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(contiguous(input))

    @doc """
    Returns a deep copy of `input`. The returned tensor has the same data
    and type but does not share storage with the original.
    """
    @spec clone(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(clone(input))

    @doc """
    Returns a new tensor detached from the current computation graph.
    The result will never require gradient.
    """
    @spec detach(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(detach(input))

    @doc """
    Returns a new tensor with the same data but of a different shape.

    The returned tensor shares the same data and must have the same
    number of elements. A single dimension may be `-1`, in which case
    it's inferred from the remaining dimensions.

    ## Args
      * `input` (`ExTorch.Tensor`) - the input tensor.
      * `shape` (`tuple`) - the desired shape.
    """
    @spec view(ExTorch.Tensor.t(), tuple()) :: ExTorch.Tensor.t()
    defbinding(view(input, shape))

    @doc """
    Returns a new view of the tensor with singleton dimensions expanded
    to a larger size. Pass `-1` for dimensions you don't want to change.

    ## Args
      * `input` (`ExTorch.Tensor`) - the input tensor.
      * `shape` (`tuple`) - the desired expanded size.
    """
    @spec expand(ExTorch.Tensor.t(), tuple()) :: ExTorch.Tensor.t()
    defbinding(expand(input, shape))

    @doc """
    Applies the Softmax function along `dim`: $Softmax(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}$.

    ## Args
      * `input` (`ExTorch.Tensor`) - the input tensor.
      * `dim` (`integer`) - the dimension along which Softmax is computed.
    """
    @spec functional_softmax(ExTorch.Tensor.t(), integer()) :: ExTorch.Tensor.t()
    defbinding(functional_softmax(input, dim))

    @doc """
    Applies LogSoftmax along `dim`: $LogSoftmax(x_i) = \\log\\left(\\frac{e^{x_i}}{\\sum_j e^{x_j}}\\right)$.

    ## Args
      * `input` (`ExTorch.Tensor`) - the input tensor.
      * `dim` (`integer`) - the dimension along which LogSoftmax is computed.
    """
    @spec functional_log_softmax(ExTorch.Tensor.t(), integer()) :: ExTorch.Tensor.t()
    defbinding(functional_log_softmax(input, dim))

    @doc """
    Applies the rectified linear unit function element-wise: $ReLU(x) = \\max(0, x)$.
    """
    @spec functional_relu(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(functional_relu(input))

    @doc """
    Sums the product of the elements of the input tensors along dimensions
    specified using a notation based on the Einstein summation convention.

    ## Args
      * `equation` (`String`) - the subscripts for the Einstein summation.
      * `a` (`ExTorch.Tensor`) - first input tensor.
      * `b` (`ExTorch.Tensor`) - second input tensor.

    ## Examples

        # Matrix multiply
        ExTorch.einsum("ij,jk->ik", a, b)

        # Batch matrix multiply
        ExTorch.einsum("bij,bjk->bik", a, b)

        # Dot product
        ExTorch.einsum("i,i->", a, b)
    """
    @spec einsum(String.t(), ExTorch.Tensor.t(), ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(einsum(equation, a, b))
  end
end
