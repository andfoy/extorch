defmodule ExTorch.Native.Tensor.Ops.Manipulation do
  @moduledoc false

  use ExTorch.Native.BindingDeclaration

  defbindings(:tensor_manipulation) do
    @doc """
    Append an empty dimension to a tensor on a given dimension.

    ## Arguments
      - `tensor`: Input tensor (`ExTorch.Tensor`)
      - `dim`: Dimension (`integer()`)

    ## Examples
        iex> x = ExTorch.full({2, 2}, -2)
        #Tensor<
        [[-2., -2.],
         [-2., -2.]]
        [
          size: {2, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.unsqueeze(x, -1)
        #Tensor<
        [[[-2.],
          [-2.]],

         [[-2.],
          [-2.]]]
        [
          size: {2, 2, 1},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.unsqueeze(x, 1)
        #Tensor<
        [[[-2., -2.]],

         [[-2., -2.]]]
        [
          size: {2, 1, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.unsqueeze(x, 0)
        #Tensor<
        [[[-2., -2.],
          [-2., -2.]]]
        [
          size: {1, 2, 2},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

    """
    @spec unsqueeze(
            ExTorch.Tensor.t(),
            integer()
          ) :: ExTorch.Tensor.t()
    defbinding(unsqueeze(tensor, dim))

    @doc """
    Returns a tensor with the same data and number of elements as `input`, but
    with the specified shape.

    When possible, the returned tensor will be a view of `input`. Otherwise, it
    will be a copy. Contiguous inputs and inputs with compatible strides can be
    reshaped without copying, but you should not depend on the copying vs.
    viewing behavior.

    A single dimension may be -1, in which case it’s inferred from the\
    remaining dimensions and the number of elements in input.

    ## Arguments
      - `tensor`: input tensor (`ExTorch.Tensor`)
      - `shape`: the new shape (`[integer()] | tuple()`)

    ## Examples
        iex> a = ExTorch.arange(0, 20) |> ExTorch.reshape({5, 4})
        #Tensor<
        [[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]
        [
          size: {5, 4},
          dtype: :double,
          device: :cpu,
          requires_grad: false
        ]>

        iex> b = ExTorch.tensor([[0, 1], [2, 3]]) |> ExTorch.reshape({-1})
        #Tensor<
        [0, 1, 2, 3]
        [
          size: {4},
          dtype: :byte,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec reshape(ExTorch.Tensor.t(), tuple() | [integer()]) :: ExTorch.Tensor.t()
    defbinding(reshape(tensor, shape))

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

    @doc """
    Returns a view of the tensor conjugated and with the last two dimensions transposed.

    `ExTorch.adjoint(x)` is equivalent to `ExTorch.conj(ExTorch.transpose(x, -2, -1))` and to
    `ExTorch.transpose(x, -2, -1)` for real tensors.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Examples
        iex> x = ExTorch.arange(4)
        iex> a = ExTorch.complex(x, x) |> ExTorch.reshape({2, 2})
        #Tensor<
        [[0.0000+0.0000j, 1.0000+1.0000j],
         [2.0000+2.0000j, 3.0000+3.0000j]]
        [
          size: {2, 2},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.adjoint(a)
        #Tensor<
        [[0.0000-0.0000j, 2.0000-2.0000j],
         [1.0000-1.0000j, 3.0000-3.0000j]]
        [
          size: {2, 2},
          dtype: :complex_float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec adjoint(ExTorch.Tensor.t()) :: ExTorch.Tensor.t()
    defbinding(adjoint(input))

    @doc """
    Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dim0` (`integer()`) - the first dimension to transpose.
    - `dim1` (`integer()`) - the second dimension to transpose.

    ## Notes
    * If `input` is a strided tensor then the resulting out tensor shares its underlying storage with the
    `input` tensor, so changing the content of one would change the content of the other.
    * If `input` is a sparse tensor then the resulting out tensor does not share the underlying storage with the input tensor.
    * If input is a sparse tensor with compressed layout (SparseCSR, SparseBSR, SparseCSC or SparseBSC) the arguments
    `dim0` and `dim1` must be both batch dimensions, or must both be sparse dimensions.
    The batch dimensions of a sparse tensor are the dimensions preceding the sparse dimensions.

    ## Examples
        iex> a = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
        #Tensor<
        [[0.0000, 1.0000, 2.0000],
         [3.0000, 4.0000, 5.0000]]
        [
          size: {2, 3},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.transpose(a, 0, 1)
        #Tensor<
        [[0.0000, 3.0000],
         [1.0000, 4.0000],
         [2.0000, 5.0000]]
        [
          size: {3, 2},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec transpose(ExTorch.Tensor.t(), integer(), integer()) :: ExTorch.Tensor.t()
    defbinding(transpose(input, dim0, dim1))

    @doc """
    Concatenates the given sequence of `seq` tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.

    ## Arguments
    - `tensors` (`[ExTorch.Tensor] | tuple()`) - A sequence of tensors of the same type. Non-empty
    tensors provided must have the same shape, except in the cat dimension.

    ## Optional arguments
    - `dim` (`integer()`) - the dimension over which the tensors are concatenated. Default: 0
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to store the
    concatenation output. Default: nil

    ## Examples
        iex> x = ExTorch.arange(5) |> ExTorch.unsqueeze(-1)
        #Tensor<
        [[0.0000],
         [1.0000],
         [2.0000],
         [3.0000],
         [4.0000]]
        [
          size: {5, 1},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>

        iex> ExTorch.cat([x, x], -1)
        #Tensor<
        [[0.0000, 0.0000],
         [1.0000, 1.0000],
         [2.0000, 2.0000],
         [3.0000, 3.0000],
         [4.0000, 4.0000]]
        [
          size: {5, 2},
          dtype: :float,
          device: :cpu,
          requires_grad: false
        ]>
    """
    @spec cat([ExTorch.Tensor.t()] | tuple(), integer(), ExTorch.Tensor.t() | nil) ::
            ExTorch.Tensor.t()
    defbinding(cat(input, dim \\ 0, out \\ nil), fn_aliases: [:concatenate, :concat])

    @doc """
    Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.

    If the tensor size along the given dimension `dim` is divisible by `chunks`, all returned chunks
    will be the same size. If the tensor size along the given dimension `dim` is not divisible by `chunks`,
    all returned chunks will be the same size, except the last one. If such division is not possible,
    this function may return fewer than the specified number of chunks.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the tensor to split
    - `chunks` (`integer`) - number of chunks to return

    ## Optional arguments
    - `dim` (`integer`) - dimension along which to split the tensor. Default: 0

    ## Notes
    * This function may return fewer than the specified number of chunks!
    * Use `ExTorch.tensor_split/3` to ensure that the result will have the exact number of chunks.

    ## Examples
        iex> ExTorch.arange(11) |> ExTorch.chunk(6)
        [
          #Tensor<
        [0.0000, 1.0000]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [2., 3.]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [4., 5.]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [6., 7.]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [8., 9.]
        [
            size: {2},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [10.]
        [size: {1}, dtype: :float, device: :cpu, requires_grad: false]>
        ]
    """
    @spec chunk(ExTorch.Tensor.t(), integer(), integer()) :: [ExTorch.Tensor.t()]
    defbinding(chunk(input, chunks, dim \\ 0))

    @doc """
    Splits a tensor into multiple sub-tensors, all of which are views of `input`,
    along dimension `dim` according to the indices or number of sections specified
    by `indices_or_sections`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the tensor to split.
    - `indices_or_sections` (`integer | ExTorch.Tensor | [integer()] | tuple`) -
      * If `indices_or_sections` is an integer `n` or a zero dimensional long tensor with value `n`,
      `input` is split into `n` sections along dimension `dim`. If `input` is divisible by `n`
      along dimension `dim`, each section will be of equal size, `input.size[dim] / n`.
      * If `input` is not divisible by n, the sizes of the first `input.size[dim] % n` sections will have size
      `input.size[dim] / n + 1`, and the rest will have size `input.size[dim] / n`.
      * If `indices_or_sections` is a list or tuple of ints, or a one-dimensional long tensor,
      then `input` is split along dimension `dim` at each of the indices in the list, tuple or tensor.
      For instance, `indices_or_sections = [2, 3]` and `dim = 0` would result in the tensors
      `input[:2]`, `input[2:3]`, and `input[3:]`.
      * If `indices_or_sections` is a tensor, it must be a zero-dimensional or one-dimensional long tensor on the CPU.

    ## Optional arguments
    - `dim` (`integer`) - dimension along which to split the tensor. Default: 0

    ## Examples
        # Split a tensor in a given number of chunks
        iex> a = ExTorch.arange(10)
        iex> ExTorch.tensor_split(a, 2)
        [
          #Tensor<
        [0.0000, 1.0000, 2.0000, 3.0000, 4.0000]
        [
            size: {5},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>,
          #Tensor<
        [5., 6., 7., 8., 9.]
        [
            size: {5},
            dtype: :float,
            device: :cpu,
            requires_grad: false
          ]>
        ]

        # Split a tensor into the given sections
        iex> ExTorch.tensor_split(a, [2, 5])
        [
          #Tensor<
          [0.0000, 1.0000]
          [size: {2}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          [2., 3., 4.]
          [size: {3}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          [5., 6., 7., 8., 9.]
          [size: {5}, dtype: :float, device: :cpu, requires_grad: false]>
        ]

    """
    @spec tensor_split(
            ExTorch.Tensor.t(),
            integer() | [integer()] | tuple() | ExTorch.Tensor.t(),
            integer()
          ) :: [ExTorch.Tensor.t()]
    defbinding(tensor_split(input, indices_or_sections, dim \\ 0),
      indices_or_sections:
        case indices_or_sections do
          x when is_integer(x) -> x
          %ExTorch.Tensor{} = x -> x
          x -> ExTorch.tensor(x, device: :cpu, dtype: :int64)
        end
    )

    @doc """
    Splits `input`, a tensor with three or more dimensions, into multiple tensors
    depthwise according to `indices_or_sections`. Each split is a view of `input`.

    This is equivalent to calling `ExTorch.tensor_split(input, indices_or_sections, dim: 2)`
    (the split dimension is 2), except that if `indices_or_sections` is an integer it must
    evenly divide the split dimension or a runtime error will be thrown.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - tensor to split.
    - `indices_or_sections` (`integer() | [integer()] | tuple()`) - See argument in `ExTorch.tensor_split/3`

    ## Examples
        iex> t = ExTorch.arange(16) |> ExTorch.reshape({2, 2, 4})
        #Tensor<
        [[[ 0.0000,  1.0000,  2.0000,  3.0000],
          [ 4.0000,  5.0000,  6.0000,  7.0000]],

         [[ 8.0000,  9.0000, 10.0000, 11.0000],
          [12.0000, 13.0000, 14.0000, 15.0000]]]
        [size: {2, 2, 4}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.dsplit(t, 2)
        [
          #Tensor<
          [[[ 0.0000,  1.0000],
            [ 4.0000,  5.0000]],

           [[ 8.0000,  9.0000],
            [12.0000, 13.0000]]]
          [size: {2, 2, 2}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          [[[ 2.,  3.],
            [ 6.,  7.]],

           [[10., 11.],
            [14., 15.]]]
          [size: {2, 2, 2}, dtype: :float, device: :cpu, requires_grad: false]>
        ]

    """
    @spec dsplit(ExTorch.Tensor.t(), integer() | [integer()] | tuple()) :: [
            ExTorch.Tensor.t()
          ]
    defbinding(dsplit(input, indices_or_sections))

    @doc """
    Creates a new tensor by horizontally stacking the tensors in `tensors`.

    Equivalent to `ExTorch.hstack(tensors)`, except each zero or one dimensional
    tensor `t` in `tensors` is first reshaped into a `(ExTorch.Tensor.numel(t), 1)`
    column before being stacked horizontally.

    ## Arguments
    - tensors (`[ExTorch.Tensor] | tuple()`) - sequence of tensors to concatenate.

    ## Optional arguments
    - out (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        # Stack two 1D tensors
        iex> a = ExTorch.tensor([1, 2, 3])
        iex> b = ExTorch.tensor([4, 5, 6])
        iex> ExTorch.column_stack([a, b])
        #Tensor<
        [[1, 4],
         [2, 5],
         [3, 6]]
        [size: {3, 2}, dtype: :byte, device: :cpu, requires_grad: false]>

        # Stack 2D tensors
        iex> a = ExTorch.arange(5)
        iex> b = ExTorch.arange(10) |> ExTorch.reshape({5, 2})
        iex> ExTorch.column_stack({a, b, b})
        #Tensor<
        [[0.0000, 0.0000, 1.0000, 0.0000, 1.0000],
         [1.0000, 2.0000, 3.0000, 2.0000, 3.0000],
         [2.0000, 4.0000, 5.0000, 4.0000, 5.0000],
         [3.0000, 6.0000, 7.0000, 6.0000, 7.0000],
         [4.0000, 8.0000, 9.0000, 8.0000, 9.0000]]
        [size: {5, 5}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec column_stack([ExTorch.Tensor.t()] | tuple(), ExTorch.Tensor.t() | nil) ::
            ExTorch.Tensor.t()
    defbinding(column_stack(tensors, out \\ nil))

    @doc """
    Stack tensors in sequence depthwise (along third axis).

    This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped
    by `ExTorch.atleast_3d/1`.

    ## Arguments
    - `tensors` (`[ExTorch.Tensor.t()] | tuple()`) - sequence of tensors to concatenate.

    ## Optional arguments
     - out (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        iex> a = ExTorch.tensor([1, 2, 3])
        iex> b = ExTorch.tensor([4, 5, 6])
        iex> ExTorch.dstack({a, b})
        #Tensor<
        [[[1, 4],
          [2, 5],
          [3, 6]]]
        [size: {1, 3, 2}, dtype: :byte, device: :cpu, requires_grad: false]>

        iex> a = ExTorch.tensor([[1],[2],[3]])
        #Tensor<
        [[1],
         [2],
         [3]]
        [size: {3, 1}, dtype: :byte, device: :cpu, requires_grad: false]>
        iex> b = ExTorch.tensor([[4],[5],[6]])
        #Tensor<
        [[4],
         [5],
         [6]]
        [size: {3, 1}, dtype: :byte, device: :cpu, requires_grad: false]>
        iex> ExTorch.dstack([a, b])
        #Tensor<
        [[[1, 4]],

         [[2, 5]],

         [[3, 6]]]
        [size: {3, 1, 2}, dtype: :byte, device: :cpu, requires_grad: false]>
    """
    @spec dstack([ExTorch.Tensor.t()] | tuple(), ExTorch.Tensor.t() | nil) :: ExTorch.Tensor.t()
    defbinding(dstack(tensors, out \\ nil))

    @doc """
    Gathers values along an axis specified by dim.

    For a 3-D tensor the output is specified by:

    ```
    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    ```

    `input` and `index` must have the same number of dimensions. It is also required
    that `ExTorch.Tensor.size(index, d) <= ExTorch.Tensor.size(input, d)` for all dimensions `d != dim`.
    `out` will have the same shape as index. Note that `input` and `index` do not broadcast
    against each other.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the source tensor.
    - `dim` (`integer()`) - the axis along which to index.
    - `index` (`ExTorch.Tensor`) - the indices of elements to gather. Its dtype must be `:int64` or `:long`

    ## Optional arguments
    - `sparse_grad` (`boolean()`) - if `true`, then the gradient w.r.t. `input` will be a sparse tensor. Default: `false`
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        iex> t = ExTorch.tensor([[1, 2], [3, 4]])
        iex> ExTorch.gather(t, 1, ExTorch.tensor([[0, 0], [1, 0]], dtype: :int64))
        #Tensor<
        [[1, 1],
         [4, 3]]
        [size: {2, 2}, dtype: :byte, device: :cpu, requires_grad: false]>
    """
    @spec gather(
            ExTorch.Tensor.t(),
            integer(),
            ExTorch.Tensor.t(),
            boolean(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(gather(input, dim, index, sparse_grad \\ false, out \\ nil))

    @doc """
    Splits `input`, a tensor with one or more dimensions, into multiple tensors horizontally according to `indices_or_sections`.
    Each split is a view of `input`.

    If `input` is one dimensional this is equivalent to calling `ExTorch.tensor_split(input, indices_or_sections, dim: 0)`
    (the split dimension is zero), and if `input` has two or more dimensions it’s equivalent to calling
    `ExTorch.tensor_split(input, indices_or_sections, dim: 1)` (the split dimension is 1),
    except that if `indices_or_sections` is an integer it must evenly divide the split
    dimension or a runtime error will be thrown.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - tensor to split.
    - `indices_or_sections` (`integer() | [integer()] | tuple()`) - See argument in `ExTorch.tensor_split/3`

    ## Examples
        iex> a = ExTorch.arange(16) |> ExTorch.reshape({4, 4})
        #Tensor<
        [[ 0.0000,  1.0000,  2.0000,  3.0000],
         [ 4.0000,  5.0000,  6.0000,  7.0000],
         [ 8.0000,  9.0000, 10.0000, 11.0000],
         [12.0000, 13.0000, 14.0000, 15.0000]]
        [size: {4, 4}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.hsplit(a, 2)
        [
          #Tensor<
          [[ 0.0000,  1.0000],
           [ 4.0000,  5.0000],
           [ 8.0000,  9.0000],
           [12.0000, 13.0000]]
          [size: {4, 2}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          [[ 2.,  3.],
           [ 6.,  7.],
           [10., 11.],
           [14., 15.]]
          [size: {4, 2}, dtype: :float, device: :cpu, requires_grad: false]>
        ]

        iex> ExTorch.hsplit(a, [3, 6])
        [
          #Tensor<
          [[ 0.0000,  1.0000,  2.0000],
           [ 4.0000,  5.0000,  6.0000],
           [ 8.0000,  9.0000, 10.0000],
           [12.0000, 13.0000, 14.0000]]
          [size: {4, 3}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          [[ 3.],
           [ 7.],
           [11.],
           [15.]]
          [size: {4, 1}, dtype: :float, device: :cpu, requires_grad: false]>,
          #Tensor<
          []
          [size: {4, 0}, dtype: :float, device: :cpu, requires_grad: false]>
        ]
    """
    @spec hsplit(ExTorch.Tensor.t(), integer() | [integer()] | tuple()) :: [ExTorch.Tensor.t()]
    defbinding(hsplit(input, indices_or_sections))

    @doc """
    Stack tensors in sequence horizontally (column wise).

    This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for all other tensors.

    ## Arguments
    - `tensors` (`[ExTorch.Tensor] | tuple()`) - sequence of tensors to concatenate.

    ## Optional arguments
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        iex> a = ExTorch.tensor([1, 2, 3])
        iex> b = ExTorch.tensor([4, 5, 6])
        iex> ExTorch.hstack({a, b})
        #Tensor<
        [1, 2, 3, 4, 5, 6]
        [size: {6}, dtype: :byte, device: :cpu, requires_grad: false]>

        iex> a = ExTorch.tensor([[1],[2],[3]])
        #Tensor<
        [[1],
         [2],
         [3]]
        [size: {3, 1}, dtype: :byte, device: :cpu, requires_grad: false]>
        iex> b = ExTorch.tensor([[4],[5],[6]])
        #Tensor<
        [[4],
         [5],
         [6]]
        [size: {3, 1}, dtype: :byte, device: :cpu, requires_grad: false]>

        iex> ExTorch.hstack([a, b])
        #Tensor<
        [[1, 4],
         [2, 5],
         [3, 6]]
        [size: {3, 2}, dtype: :byte, device: :cpu, requires_grad: false]>
    """
    @spec hstack([ExTorch.Tensor.t()] | tuple(), ExTorch.Tensor.t() | nil) :: ExTorch.Tensor.t()
    defbinding(hstack(tensors, out \\ nil))

    @doc """
    Moves the dimension(s) of `input` at the position(s) in `source` to the position(s) in `destination`.

    Other dimensions of `input` that are not explicitly moved remain in their original order and appear
    at the positions not specified in `destination`.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `source` (`integer() | tuple()`) - original positions of the dims to move. These must be unique.
    - `destination` (`integer() | tuple()`) - destination positions of the dims to move. These must be unique.

    ## Examples
        iex> a = ExTorch.randn({3, 2, 1})
        #Tensor<
        [[[-0.0404],
          [ 0.5073]],

         [[ 0.3008],
          [-0.6428]],

         [[-0.8649],
          [ 0.3615]]]
        [size: {3, 2, 1}, dtype: :float, device: :cpu, requires_grad: false]>

        # Swap two singular dimensions
        iex> ExTorch.movedim(a, 1, 0)
        #Tensor<
        [[[-0.0404],
          [ 0.3008],
          [-0.8649]],

         [[ 0.5073],
          [-0.6428],
          [ 0.3615]]]
        [size: {2, 3, 1}, dtype: :float, device: :cpu, requires_grad: false]>

        # Swap multiple dimensions
        iex> ExTorch.movedim(a, {1, 2}, {0, 1})
        #Tensor<
        [[[-0.0404,  0.3008, -0.8649]],

         [[ 0.5073, -0.6428,  0.3615]]]
        [size: {2, 1, 3}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec movedim(ExTorch.Tensor.t(), tuple() | integer(), tuple() | integer()) ::
            ExTorch.Tensor.t()
    defbinding(movedim(input, source, destination), fn_aliases: [:moveaxis])

    @doc """
    Returns a new tensor that is a narrowed version of `input` tensor.

    The dimension `dim` is input from `start` to `start` + `length`.
    The returned tensor and `input` tensor share the same underlying storage.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the tensor to narrow.
    - `dim` (`integer`) - the dimension along which to narrow.
    - `start` (`integer | ExTorch.Tensor`) - index of the element to start the
    narrowed dimension from. Can be negative, which means indexing from the
    end of `dim`. If `ExTorch.Tensor`, it must be an 0-dim integral Tensor (bools not allowed).
    - `length` (`integer`) - length of the narrowed dimension, must be weakly positive.

    ## Examples
        iex> a = ExTorch.arange(12) |> ExTorch.reshape({4, 3})
        #Tensor<
        [[ 0.0000,  1.0000,  2.0000],
         [ 3.0000,  4.0000,  5.0000],
         [ 6.0000,  7.0000,  8.0000],
         [ 9.0000, 10.0000, 11.0000]]
        [size: {4, 3}, dtype: :float, device: :cpu, requires_grad: false]>

        # Narrow tensor from 0 to 2 in the first dimension
        iex> ExTorch.narrow(a, 0, 0, 2)
        #Tensor<
        [[0.0000, 1.0000, 2.0000],
         [3.0000, 4.0000, 5.0000]]
        [size: {2, 3}, dtype: :float, device: :cpu, requires_grad: false]>

        # Narrow tensor from 1 to 3 in the second dimension
        iex> ExTorch.narrow(a, 1, 1, 2)
        #Tensor<
        [[ 1.,  2.],
         [ 4.,  5.],
         [ 7.,  8.],
         [10., 11.]]
        [size: {4, 2}, dtype: :float, device: :cpu, requires_grad: false]>

        # Narrow tensor using a `start` tensor
        iex> ExTorch.narrow(a, -1, ExTorch.tensor(-1), 1)
        #Tensor<
        [[ 2.],
         [ 5.],
         [ 8.],
         [11.]]
        [size: {4, 1}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec narrow(ExTorch.Tensor.t(), integer(), integer() | ExTorch.Tensor.t(), integer()) ::
            ExTorch.Tensor.t()
    defbinding(narrow(input, dim, start, length))

    @doc """
    Same as `ExTorch.narrow/4` except this returns a copy rather than shared storage.
    This is primarily for sparse tensors, which do not have a shared-storage narrow method.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the tensor to narrow.
    - `dim` (`integer`) - the dimension along which to narrow.
    - `start` (`integer`) - index of the element to start the
    narrowed dimension from. Can be negative, which means indexing from the
    end of `dim`.
    - `length` (`integer`) - length of the narrowed dimension, must be weakly positive.

    ## Optional arguments
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        iex> a = ExTorch.arange(12) |> ExTorch.reshape({4, 3})
        #Tensor<
        [[ 0.0000,  1.0000,  2.0000],
         [ 3.0000,  4.0000,  5.0000],
         [ 6.0000,  7.0000,  8.0000],
         [ 9.0000, 10.0000, 11.0000]]
        [size: {4, 3}, dtype: :float, device: :cpu, requires_grad: false]>

        # Narrow tensor from 0 to 2 in the first dimension
        iex> ExTorch.narrow_copy(a, 0, 0, 2)
        #Tensor<
        [[0.0000, 1.0000, 2.0000],
         [3.0000, 4.0000, 5.0000]]
        [size: {2, 3}, dtype: :float, device: :cpu, requires_grad: false]>

        # Narrow tensor from 1 to 3 in the second dimension
        iex> ExTorch.narrow_copy(a, 1, 1, 2)
        #Tensor<
        [[ 1.,  2.],
         [ 4.,  5.],
         [ 7.,  8.],
         [10., 11.]]
        [size: {4, 2}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec narrow_copy(
            ExTorch.Tensor.t(),
            integer(),
            integer(),
            integer(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(narrow_copy(input, dim, start, length, out \\ nil))

    @doc """
    Retrieve the indices of all non-zero elements in a tensor.

    This function can behave differently depending of the value set by the
    `as_tuple` parameter:

    ### When `as_tuple` is `false` (default):
    Returns a tensor containing the indices of all non-zero elements of `input`.
    Each row in the result contains the indices of a non-zero element in `input`.
    The result is sorted lexicographically, with the last index changing the fastest (C-style).

    If `input` has $n$ dimensions, then the resulting indices tensor out is of size $(z \\times n)$,
    where $z$ is the total number of non-zero elements in the input tensor.

    ### When `as_tuple` is `true`
    Returns a tuple of 1-D tensors, one for each dimension in `input`, each containing the indices
    (in that dimension) of all non-zero elements of input .

    If `input` has $n$ dimensions, then the resulting tuple contains $n$ tensors of size $z$, where $z$
    is the total number of non-zero elements in the input tensor.

    As a special case, when `input` has zero dimensions and a nonzero scalar value, it is treated as a
    one-dimensional tensor with one element.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.

    ## Optional arguments
    - `out` (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
      store the output result. This will only take effect when `as_tuple = false`.
      Default: `nil`
    - `as_tuple` (`boolean`) - if `false`, the function will return the output tensor
    containing indices. Else, it returns one 1-D tensor for each dimension, containing
    the indices of each nonzero element along that dimension. Default: `false`

    ## Examples
        iex> input1 = ExTorch.tensor([1, 1, 1, 0, 1])
        iex> input2 = ExTorch.tensor([[0.6, 0.0, 0.0, 0.0],
        ...>                          [0.0, 0.4, 0.0, 0.0],
        ...>                          [0.0, 0.0, 1.2, 0.0],
        ...>                          [0.0, 0.0, 0.0,-0.4]])

        # Return tensor indices
        iex> ExTorch.nonzero(input1)
        #Tensor<
        [[0],
         [1],
         [2],
         [4]]
        [size: {4, 1}, dtype: :long, device: :cpu, requires_grad: false]>

        iex> ExTorch.nonzero(input2)
        #Tensor<
        [[0, 0],
         [1, 1],
         [2, 2],
         [3, 3]]
        [size: {4, 2}, dtype: :long, device: :cpu, requires_grad: false]>

        # Return tuple indices
        iex> ExTorch.nonzero(input1, as_tuple: true)
        #Tensor<
        [0, 1, 2, 4]
        [size: {4}, dtype: :long, device: :cpu, requires_grad: false]>

        iex> ExTorch.nonzero(input2, as_tuple: true)
        {#Tensor<
         [0, 1, 2, 3]
         [size: {4}, dtype: :long, device: :cpu, requires_grad: false]>,
         #Tensor<
         [0, 1, 2, 3]
         [size: {4}, dtype: :long, device: :cpu, requires_grad: false]>}
    """
    @spec nonzero(ExTorch.Tensor.t(), ExTorch.Tensor.t() | nil, boolean()) ::
            ExTorch.Tensor.t() | tuple()
    defbinding(nonzero(input, out \\ nil, as_tuple \\ false))

    @doc """
    Returns a view of the original tensor `input` with its dimensions permuted.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dims` (`tuple() | [integer()]`) - The desired ordering of dimensions.

    ## Examples
        iex> a = ExTorch.rand({3, 2, 4, 5})
        iex> out = ExTorch.permute(a, {2, -1, 0, 1})
        iex> out.size
        {4, 5, 3, 2}
    """
    @spec permute(ExTorch.Tensor.t(), tuple() | [integer()]) :: ExTorch.Tensor.t()
    defbinding(permute(input, dims))

    @doc """
    Stack tensors in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after all 1-D tensors
    have been reshaped by `ExTorch.atleast_2d/1`.

    ## Arguments
    - `tensors` (`[ExTorch.Tensor.t()] | tuple()`) - sequence of tensors to concatenate.

    ## Optional arguments
     - out (`ExTorch.Tensor | nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        iex> a = ExTorch.tensor([1, 2, 3])
        iex> b = ExTorch.tensor([4, 5, 6])
        iex> ExTorch.vstack({a, b})
        #Tensor<
        [[1, 2, 3],
         [4, 5, 6]]
        [size: {2, 3}, dtype: :byte, device: :cpu, requires_grad: false]>

        iex> a = ExTorch.tensor([[1],[2],[3]])
        #Tensor<
        [[1],
         [2],
         [3]]
        [size: {3, 1}, dtype: :byte, device: :cpu, requires_grad: false]>
        iex> b = ExTorch.tensor([[4],[5],[6]])
        #Tensor<
        [[4],
         [5],
         [6]]
        [size: {3, 1}, dtype: :byte, device: :cpu, requires_grad: false]>

        iex> ExTorch.vstack([a, b])
        #Tensor<
        [[1],
         [2],
         [3],
         [4],
         [5],
         [6]]
        [size: {6, 1}, dtype: :byte, device: :cpu, requires_grad: false]>

    """
    @spec vstack([ExTorch.Tensor.t()] | tuple(), ExTorch.Tensor.t() | nil) :: ExTorch.Tensor.t()
    defbinding(vstack(tensors, out \\ nil), fn_aliases: [:row_stack])

    @doc """
    Writes all values from the tensor `src` into `input` at the indices specified in the `index` tensor.
    For each value in `src`, its output index is specified by its index in `src` for `dimension != dim` and by
    the corresponding value in `index` for `dimension = dim`.

    For a 3-D tensor, `input` is updated as:

    ```
    input[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    input[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    input[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    ```

    This is the reverse operation of the manner described in `ExTorch.gather/5`.

    `input`, `index` and `src` (if it is a `ExTorch.Tensor`) should all have the same number of dimensions.
    It is also required that `index.size(d) <= src.size(d)` for all dimensions `d`, and that
    `index.size(d) <= input.size(d)` for all `dimensions d != dim`. Note that `index` and `src` do not broadcast.

    Moreover, as for `ExTorch.gather/5`, the values of `index` must be between 0 and `input.size(dim) - 1`
    inclusive.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `dim` (`integer`) - the axis along which to index.
    - `index` (`ExTorch.Tensor`) -  the indices of elements to scatter, can be either empty or of the
    same dimensionality as `src`. When empty, the operation returns `input` unchanged. It's dtype must be `:long` or
    `:int64`
    - `src` (`ExTorch.Tensor` or `float`) - the source element(s) to scatter.

    ## Optional arguments
    - `out` (`ExTorch.Tensor` or `nil`) - an optional pre-allocated tensor used to
    store the output result. If `inplace = true` it will not take any effect. Default: `nil`
    - `inplace` (`bool`) - if `true` then the scatter operation will take inplace on the `input` argument. Else it will
    return a separate tensor with the result. Default: `false`

    ## Warnings
    When indices are not unique, the behavior is non-deterministic (one of the values from `src` will be picked arbitrarily)
    and the gradient will be incorrect (it will be propagated to all locations in the source that correspond to the same index)!

    ## Notes
    1. The backward pass is implemented only for `src.size == index.size`.
    2. This function does not expose the `reduce` argument since it is set for deprecation. It is recommended to use
    the `ExTorch.scatter_reduce` function instead.

    ## Examples
        iex> src = ExTorch.arange(1, 11) |> ExTorch.reshape({2, 5})
        #Tensor<
        [[ 1.,  2.,  3.,  4.,  5.],
         [ 6.,  7.,  8.,  9., 10.]]
        [size: {2, 5}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> index = ExTorch.tensor([[0, 1, 2, 0]], dtype: :int64)
        #Tensor<
        [[0, 1, 2, 0]]
        [size: {1, 4}, dtype: :long, device: :cpu, requires_grad: false]>
        iex> input = ExTorch.zeros({3, 5}, dtype: src.dtype)
        #Tensor<
        [[   0.,    0.,    0.,    0.,    0.],
         [   0.,    0.,    0.,    0.,    0.],
         [   0.,    0.,    0.,    0.,    0.]]
        [size: {3, 5}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.scatter(input, 0, index, src)
        #Tensor<
        [[1.0000, 0.0000, 0.0000, 4.0000, 0.0000],
         [0.0000, 2.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 3.0000, 0.0000, 0.0000]]
        [size: {3, 5}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> index = ExTorch.tensor([[0, 1, 2], [0, 1, 4]], dtype: :int64)
        #Tensor<
        [[0, 1, 2],
         [0, 1, 4]]
        [size: {2, 3}, dtype: :long, device: :cpu, requires_grad: false]>
        iex> ExTorch.scatter(input, 1, index, src)
        #Tensor<
        [[1.0000, 2.0000, 3.0000, 0.0000, 0.0000],
         [6.0000, 7.0000, 0.0000, 0.0000, 8.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]
        [size: {3, 5}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec scatter(
            ExTorch.Tensor.t(),
            integer(),
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t() | float(),
            ExTorch.Tensor.t() | nil,
            boolean()
          ) :: ExTorch.Tensor.t()
    defbinding(scatter(input, dim, index, src, out \\ nil, inplace \\ false),
      src:
        case is_float(src) do
          true -> ExTorch.tensor(src)
          false -> src
        end
    )

    @doc """
    Embeds the values of the `src` tensor into `input` along the diagonal elements of `input`,
    with respect to `dim1` and `dim2`.

    This function returns a tensor with fresh storage; it does not return a view.

    The argument `offset` controls which diagonal to consider:

    * If `offset = 0`, it is the main diagonal.
    * If `offset > 0`, it is above the main diagonal.
    * If `offset < 0`, it is below the main diagonal.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor. Must be at least 2-dimensional.
    - `src` (`ExTorch.Tensor`) - the tensor to embed into `input`.
    - `offset` (`integer`) - which diagonal to consider. Default: 0 (main diagonal).
    - `dim1` (`integer`) - first dimension with respect to which to take diagonal. Default: 0.
    - `dim2` (`integer`) - second dimension with respect to which to take diagonal. Default: 1.

    ## Optional arguments
    - `out` (`ExTorch.Tensor` or `nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Notes
    `src` must be of the proper size in order to be embedded into `input`. Specifically, it should have
    the same shape as `ExTorch.diagonal(input, offset, dim1, dim2)`

    ## Examples
        iex> a = ExTorch.zeros({3, 3})
        #Tensor<
        [[   0.,    0.,    0.],
         [   0.,    0.,    0.],
         [   0.,    0.,    0.]]
        [size: {3, 3}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.diagonal_scatter(a, ExTorch.ones(3), 0)
        #Tensor<
        [[1.0000, 0.0000, 0.0000],
         [0.0000, 1.0000, 0.0000],
         [0.0000, 0.0000, 1.0000]]
        [size: {3, 3}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.diagonal_scatter(a, ExTorch.ones(2), 1)
        #Tensor<
        [[0.0000, 1.0000, 0.0000],
         [0.0000, 0.0000, 1.0000],
         [0.0000, 0.0000, 0.0000]]
        [size: {3, 3}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec diagonal_scatter(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t(),
            integer(),
            integer(),
            integer(),
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(diagonal_scatter(input, src, offset \\ 0, dim1 \\ 0, dim2 \\ 1, out \\ nil))

    @doc """
    Embeds the values of the `src` tensor into `input` at the given `index`.
    This function returns a tensor with fresh storage; it does not create a view.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `src` (`ExTorch.Tensor`) - the tensor to embed into `input`.
    - `dim` (`integer`) - the dimension to insert the slice into.
    - `index` (`integer`) - the index to select with.

    ## Optional arguments
    - `out` (`ExTorch.Tensor` or `nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Note
    `src` must be of the proper size in order to be embedded into `input`.
    Specifically, it should have the same shape as `ExTorch.select(input, dim, index)`

    ## Examples
        iex> a = ExTorch.zeros({2, 2})
        #Tensor<
        [[   0.,    0.],
         [   0.,    0.]]
        [size: {2, 2}, dtype: :float, device: :cpu, requires_grad: false]>
        iex> b = ExTorch.ones(2)
        #Tensor<
        [1., 1.]
        [size: {2}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> ExTorch.select_scatter(a, b, 0, 0)
        #Tensor<
        [[1.0000, 1.0000],
         [0.0000, 0.0000]]
        [size: {2, 2}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec select_scatter(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t(),
            integer(),
            integer(),
            ExTorch.Tensor.t() | nil
          ) ::
            ExTorch.Tensor.t()
    defbinding(select_scatter(input, src, dim, index, out \\ nil))

    @doc """
    Embeds the values of the `src` tensor into `input` at the given dimension.
    This function returns a tensor with fresh storage; it does not create a view.

    ## Arguments
    - `input` (`ExTorch.Tensor`) - the input tensor.
    - `src` (`ExTorch.Tensor`) - the tensor to embed into `input`.

    ## Optional arguments
    - `dim` (`integer`) - the dimension to insert the slice into. Default: 0
    - `start` (`integer` or `nil`) - the start index from which the slice should be inserted into. Default: `nil`
    - `stop` (`integer` or `nil`) - the end index until which the slice is inserted. Default: `nil`
    - `step` (`integer` or `nil`) - how many elements are skipped in between slice insertions. Default: 1
    - `out` (`ExTorch.Tensor` or `nil`) - an optional pre-allocated tensor used to
    store the output result. Default: `nil`

    ## Examples
        iex> a = ExTorch.zeros({8, 8})
        iex> b = ExTorch.ones({2, 8})
        iex> ExTorch.slice_scatter(a, b, start: 6)
        #Tensor<
        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]
        [size: {8, 8}, dtype: :float, device: :cpu, requires_grad: false]>

        iex> b = ExTorch.ones({8, 2})
        iex> ExTorch.slice_scatter(a, b, dim: 1, start: 2, stop: 6, step: 2)
        #Tensor<
        [[0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000]]
        [size: {8, 8}, dtype: :float, device: :cpu, requires_grad: false]>
    """
    @spec slice_scatter(
            ExTorch.Tensor.t(),
            ExTorch.Tensor.t(),
            integer(),
            integer() | nil,
            integer() | nil,
            integer() | nil,
            ExTorch.Tensor.t() | nil
          ) :: ExTorch.Tensor.t()
    defbinding(slice_scatter(input, src, dim \\ 0, start \\ nil, stop \\ nil, step \\ 1, out \\ nil))
  end
end
