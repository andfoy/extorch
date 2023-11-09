defmodule ExTorchTest.Tensor.IndexingTest do
  use ExUnit.Case

  test "index tensor with an integer" do
    x = ExTorch.rand({3, 4, 5})
    indexed = ExTorch.index(x, 0)
    assert indexed.size == {4, 5}
  end

  test "index tensor with an integer sequence" do
    x = ExTorch.rand({3, 4, 5})
    indexed = ExTorch.index(x, [0, 1])
    assert indexed.size == {5}
  end

  test "index tensor with a slice" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, ExTorch.slice(1))
    assert indexed.size == {5, 4, 5}

    indexed = ExTorch.index(x, ExTorch.slice(nil, -1))
    assert indexed.size == {5, 4, 5}

    indexed = ExTorch.index(x, ExTorch.slice(nil, nil, 2))
    assert indexed.size == {3, 4, 5}

    indexed = ExTorch.index(x, ExTorch.slice())
    assert indexed.size == {6, 4, 5}

    indexed = ExTorch.index(x, :"::")
    assert indexed.size == {6, 4, 5}

    indexed = ExTorch.index(x, 1..6)
    assert indexed.size == {5, 4, 5}

    indexed = ExTorch.index(x, 0..6//2)
    assert indexed.size == {3, 4, 5}
  end

  test "index tensor with a sequence of slices" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, [ExTorch.slice(), ExTorch.slice(0, -2)])
    assert indexed.size == {6, 2, 5}

    indexed = ExTorch.index(x, [ExTorch.slice(1), :"::", 0..4])
    assert indexed.size == {5, 4, 4}
  end

  test "index tensor with ellipsis" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, :...)
    assert indexed.size == {6, 4, 5}

    indexed = ExTorch.index(x, :ellipsis)
    assert indexed.size == {6, 4, 5}
  end

  test "index tensor with ellipsis and other index" do
    x = ExTorch.rand({6, 4, 2, 5})
    indexed = ExTorch.index(x, [:..., 0])
    assert indexed.size == {6, 4, 2}

    indexed = ExTorch.index(x, {:..., 0})
    assert indexed.size == {6, 4, 2}

    indexed = ExTorch.index(x, [1, :ellipsis, ExTorch.slice(0, 3)])
    assert indexed.size == {4, 2, 3}
  end

  test "index tensor with nil" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, nil)
    assert indexed.size == {1, 6, 4, 5}
  end

  test "index tensor with boolean" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, true)
    assert indexed.size == {1, 6, 4, 5}

    indexed = ExTorch.index(x, false)
    assert indexed.size == {0, 6, 4, 5}
  end

  test "index tensor with a list of integers" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, [[0, 4]])
    assert indexed.size == {2, 4, 5}

    indexed = ExTorch.index(x, [[-1, -2, 3]])
    assert indexed.size == {3, 4, 5}

    indexed = ExTorch.index(x, [{5}])
    assert indexed.size == {1, 4, 5}
  end

  test "index tensor with another tensor" do
    i = ExTorch.tensor([[5, 1], [4, 1], [3, 1], [2, 1], [1, 1], [0, 1]], dtype: :int64)
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, i)
    assert indexed.size == {6, 2, 4, 5}
  end

  test "index tensor with the native Access behaviour" do
    x = ExTorch.rand({6, 4, 5})
    indexed = x[0]
    assert indexed.size == {4, 5}

    indexed = x[{0..3, 1, [0, 2]}]
    assert indexed.size == {3, 2}
  end

  test "assign a number in a tensor dim" do
    x = ExTorch.zeros({2, 3, 3})
    expected = [[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    x = ExTorch.index_put(x, 0, -1)
    assert ExTorch.Tensor.to_list(x) == expected
  end

  test "assign a list in a tensor slice" do
    x = ExTorch.zeros({3, 3})
    expected = [[0, 0, 0], [1, 2, 3], [0, 0, 0]]
    x = ExTorch.index_put(x, [1, :"::"], [1, 2, 3])
    assert ExTorch.Tensor.to_list(x) == expected
  end

  test "assign a list in a tensor slice (broadcast)" do
    x = ExTorch.zeros({2, 3, 3})
    expected = [[[0, 0, 0], [1, 2, 3], [0, 0, 0]], [[0, 0, 0], [1, 2, 3], [0, 0, 0]]]
    x = ExTorch.index_put(x, [:"::", 1], [1, 2, 3])
    assert ExTorch.Tensor.to_list(x) == expected
  end

  test "assign a tensor in a tensor dim" do
    x = ExTorch.zeros({2, 3, 3})
    value = ExTorch.eye(3)
    expected = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    x = ExTorch.index_put(x, 0, value)
    assert ExTorch.Tensor.to_list(x) == expected
  end

  test "index_add/4" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    out = ExTorch.index_add(input, 0, index, source)

    valid_add =
      indices
      |> Enum.with_index()
      |> Enum.reduce(true, fn {to_take, i}, acc ->
        acc and ExTorch.allclose(out[to_take], source[i])
      end)

    assert valid_add
  end

  test "index_add/5" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.ones({3, 3})
    expected = ExTorch.full({3, 3}, 2)
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    out = ExTorch.index_add(input, 0, index, source, 2)

    valid_add =
      indices
      |> Enum.with_index()
      |> Enum.reduce(true, fn {to_take, i}, acc ->
        acc and ExTorch.allclose(out[to_take], expected[i])
      end)

    assert valid_add
  end

  test "index_add/5 with kwargs" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.ones({3, 3})
    expected = ExTorch.full({3, 3}, -3.4)
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    out = ExTorch.index_add(input, 0, index, source, alpha: -3.4)

    valid_add =
      indices
      |> Enum.with_index()
      |> Enum.reduce(true, fn {to_take, i}, acc ->
        acc and ExTorch.allclose(out[to_take], expected[i])
      end)

    assert valid_add
  end

  test "index_add/6" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    out = ExTorch.empty_like(input)

    expected = ExTorch.zeros({5, 3})
    expected = ExTorch.index_put(expected, index, source)

    ExTorch.index_add(input, 0, index, source, 1, out)
    assert ExTorch.allclose(out, expected)
  end

  test "index_add/6 with kwargs" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.ones({3, 3})
    expected = ExTorch.full({3, 3}, 2)
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    ExTorch.index_add(input, 0, index, source, 2, inplace: true)

    valid_add =
      indices
      |> Enum.with_index()
      |> Enum.reduce(true, fn {to_take, i}, acc ->
        acc and ExTorch.allclose(input[to_take], expected[i])
      end)

    assert valid_add
  end

  test "index_add/7" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.ones({3, 3})
    expected = ExTorch.full({3, 3}, 2)
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    ExTorch.index_add(input, 0, index, source, 2, nil, true)

    valid_add =
      indices
      |> Enum.with_index()
      |> Enum.reduce(true, fn {to_take, i}, acc ->
        acc and ExTorch.allclose(input[to_take], expected[i])
      end)

    assert valid_add
  end

  test "index_copy/4" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    out = ExTorch.index_copy(input, 0, index, source)

    valid_add =
      indices
      |> Enum.with_index()
      |> Enum.reduce(true, fn {to_take, i}, acc ->
        acc and ExTorch.allclose(out[to_take], source[i])
      end)

    assert valid_add
  end

  test "index_copy/5" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)

    expected = ExTorch.index_put(input, index, source)
    out = ExTorch.empty_like(input)

    ExTorch.index_copy(input, 0, index, source, out)
    assert ExTorch.allclose(out, expected)
  end

  test "index_copy/5 with kwargs" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    expected = ExTorch.index_put(input, index, source)

    ExTorch.index_copy(input, 0, index, source, inplace: true)
    assert ExTorch.allclose(input, expected)
  end

  test "index_copy/6" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    expected = ExTorch.index_put(input, index, source)

    ExTorch.index_copy(input, 0, index, source, nil, true)
    assert ExTorch.allclose(input, expected)
  end

  test "index_reduce/5" do
    input = ExTorch.ones({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    expected = ExTorch.index_put(input, index, source)

    out = ExTorch.index_reduce(input, 0, index, source, :prod)
    assert ExTorch.allclose(out, expected)
  end

  test "index_reduce/6" do
    input = ExTorch.zeros({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    expected = ExTorch.index_put(input, index, source)

    out = ExTorch.index_reduce(input, 0, index, source, :prod, false)
    assert ExTorch.allclose(out, expected)
  end

  test "index_reduce/6 with kwargs" do
    input = ExTorch.rand({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    expected = ExTorch.index_put(input, index, source)

    out = ExTorch.index_reduce(input, 0, index, source, :mean, include_self: false)
    assert ExTorch.allclose(out, expected)
  end

  test "index_reduce/7" do
    input = ExTorch.rand({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    result = ExTorch.maximum(input[index], source)

    expected = ExTorch.index_put(input, index, result)
    out = ExTorch.empty_like(input)

    ExTorch.index_reduce(input, 0, index, source, :amax, true, out)
    assert ExTorch.allclose(out, expected)
  end

  test "index_reduce/8" do
    input = ExTorch.rand({5, 3})
    source = ExTorch.rand({3, 3})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    result = ExTorch.minimum(input[index], source)
    expected = ExTorch.index_put(input, index, result)

    ExTorch.index_reduce(input, 0, index, source, :amin, true, nil, true)
    assert ExTorch.allclose(input, expected)
  end

  test "index_select/3" do
    input = ExTorch.rand({5, 6})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    expected = input[index]

    out = ExTorch.index_select(input, 0, index)
    assert ExTorch.allclose(out, expected)
  end

  test "index_select/4" do
    input = ExTorch.rand({5, 6})
    indices = [0, 4, 2]
    index = ExTorch.tensor(indices, dtype: :long)
    expected = input[{:"::", index}]
    out = ExTorch.empty_like(expected)

    ExTorch.index_select(input, 1, index, out)
    assert ExTorch.allclose(out, expected)
  end

  test "masked_select/2" do
    input = ExTorch.randn({5, 4})
    mask = ExTorch.ge(input, 0)
    expected = input[mask]
    out = ExTorch.masked_select(input, mask)
    assert ExTorch.allclose(out, expected)
  end

  test "masked_select/3" do
    input = ExTorch.randn({5, 4})
    mask = ExTorch.ge(input, 0)
    expected = input[mask]
    out = ExTorch.empty_like(expected)

    ExTorch.masked_select(input, mask, out)
    assert ExTorch.allclose(out, expected)
  end

  test "narrow/4" do
    input = ExTorch.randn({4, 3})
    expected = input[0..2]
    out = ExTorch.narrow(input, 0, 0, 2)
    assert ExTorch.allclose(out, expected)
  end

  test "narrow/4 with tensor start" do
    input = ExTorch.randn({4, 3})
    expected = input[{:"::", 1..3}]
    out = ExTorch.narrow(input, 1, ExTorch.tensor(1), 2)
    assert ExTorch.allclose(out, expected)
  end
end
