defmodule ExTorchTest.Tensor.ComparisonTest do
  use ExUnit.Case

  test "allclose/2" do
    t1 = ExTorch.tensor([10_000.0, 1.0e-07])
    t2 = ExTorch.tensor([10_000.1, 1.0e-08])
    assert !ExTorch.allclose(t1, t2)
  end

  test "allclose/3" do
    t1 = ExTorch.tensor([10_000.0, 1.0e-08])
    t2 = ExTorch.tensor([10_000.1, 1.0e-09])
    assert ExTorch.allclose(t1, t2, 1.0e-5)
  end

  test "allclose/3 with kwargs" do
    t1 = ExTorch.tensor([10_000.0, 1.0e-08, :nan])
    t2 = ExTorch.tensor([10_000.1, 1.0e-09, :nan])
    assert ExTorch.allclose(t1, t2, equal_nan: true)
  end

  test "allclose/4" do
    t1 = ExTorch.tensor([1.0, :nan])
    t2 = ExTorch.tensor([1.0, :nan])
    assert !ExTorch.allclose(t1, t2, 1.0e-5, 1.0e-8)
  end

  test "allclose/5" do
    t1 = ExTorch.tensor([1.0, :nan])
    t2 = ExTorch.tensor([1.0, :nan])
    assert ExTorch.allclose(t1, t2, 1.0e-5, 1.0e-8, true)
  end

  test "argsort/1" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_list = [
      [3, 0, 4, 1, 5, 6, 2],
      [1, 6, 2, 3, 0, 4, 5],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    in_tensor = ExTorch.tensor(input_list)
    argsort = ExTorch.argsort(in_tensor)
    assert ExTorch.Tensor.to_list(argsort) == expected_list
  end

  test "argsort/1 with kwargs" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_list = [
      [3, 0, 4, 1, 5, 6, 2],
      [1, 6, 2, 3, 0, 4, 5],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    in_tensor = ExTorch.tensor(input_list)
    argsort = ExTorch.argsort(in_tensor, dim: -1)
    assert ExTorch.Tensor.to_list(argsort) == expected_list
  end

  test "argsort/2" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_list = [
      [0, 1, 1, 0, 0, 0, 1],
      [2, 2, 2, 1, 2, 2, 0],
      [1, 0, 0, 2, 1, 1, 2]
    ]

    in_tensor = ExTorch.tensor(input_list)
    argsort = ExTorch.argsort(in_tensor, 0)
    assert ExTorch.Tensor.to_list(argsort) == expected_list
  end

  test "argsort/2 with kwargs" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_list = [
      [1, 0, 0, 1, 1, 1, 2],
      [2, 2, 1, 2, 2, 2, 0],
      [0, 1, 2, 0, 0, 0, 1]
    ]

    in_tensor = ExTorch.tensor(input_list)
    argsort = ExTorch.argsort(in_tensor, 0, descending: true)
    assert ExTorch.Tensor.to_list(argsort) == expected_list
  end

  test "argsort/3" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_list = [
      [2, 6, 5, 1, 4, 0, 3],
      [5, 4, 0, 3, 2, 6, 1],
      [6, 5, 4, 3, 2, 1, 0]
    ]

    in_tensor = ExTorch.tensor(input_list)
    argsort = ExTorch.argsort(in_tensor, -1, true)
    assert ExTorch.Tensor.to_list(argsort) == expected_list
  end

  test "argsort/4" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_list = [
      [2, 6, 5, 1, 4, 0, 3],
      [5, 4, 0, 3, 2, 6, 1],
      [6, 5, 4, 3, 2, 1, 0]
    ]

    in_tensor = ExTorch.tensor(input_list)
    argsort = ExTorch.argsort(in_tensor, -1, true, true)
    assert ExTorch.Tensor.to_list(argsort) == expected_list
  end

  test "sort/1" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_values = [
      [-2, -1, 0, 3, 4, 5, 10],
      [-5, 1, 2, 3, 5, 7, 20],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_indices = [
      [3, 0, 4, 1, 5, 6, 2],
      [1, 6, 2, 3, 0, 4, 5],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    in_tensor = ExTorch.tensor(input_list)
    {values, indices} = ExTorch.sort(in_tensor)
    assert ExTorch.Tensor.to_list(values) == expected_values
    assert ExTorch.Tensor.to_list(indices) == expected_indices
  end

  test "sort/2 with kwargs" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_values = [
      [-2, -1, 0, 3, 4, 5, 10],
      [-5, 1, 2, 3, 5, 7, 20],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_indices = [
      [3, 0, 4, 1, 5, 6, 2],
      [1, 6, 2, 3, 0, 4, 5],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    in_tensor = ExTorch.tensor(input_list)
    {values, indices} = ExTorch.sort(in_tensor, dim: 1, descending: false)
    assert ExTorch.Tensor.to_list(values) == expected_values
    assert ExTorch.Tensor.to_list(indices) == expected_indices
  end

  test "sort/2" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_values = [
      [-1, -5, 2, -2, 0, 4, 1],
      [0, 1, 2, 3, 4, 5, 5],
      [5, 3, 10, 3, 7, 20, 6]
    ]

    expected_indices = [
      [0, 1, 1, 0, 0, 0, 1],
      [2, 2, 2, 1, 2, 2, 0],
      [1, 0, 0, 2, 1, 1, 2]
    ]

    in_tensor = ExTorch.tensor(input_list)
    {values, indices} = ExTorch.sort(in_tensor, 0)
    assert ExTorch.Tensor.to_list(values) == expected_values
    assert ExTorch.Tensor.to_list(indices) == expected_indices
  end

  test "sort/3" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_values = [
      [5, 3, 10, 3, 7, 20, 6],
      [0, 1, 2, 3, 4, 5, 5],
      [-1, -5, 2, -2, 0, 4, 1]
    ]

    expected_indices = [
      [1, 0, 0, 1, 1, 1, 2],
      [2, 2, 1, 2, 2, 2, 0],
      [0, 1, 2, 0, 0, 0, 1]
    ]

    in_tensor = ExTorch.tensor(input_list)
    {values, indices} = ExTorch.sort(in_tensor, 0, true)
    assert ExTorch.Tensor.to_list(values) == expected_values
    assert ExTorch.Tensor.to_list(indices) == expected_indices
  end

  test "sort/3 with kwargs" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_values = [
      [10, 5, 4, 3, 0, -1, -2],
      [20, 7, 5, 3, 2, 1, -5],
      [6, 5, 4, 3, 2, 1, 0]
    ]

    expected_indices = [
      [2, 6, 5, 1, 4, 0, 3],
      [5, 4, 0, 3, 2, 6, 1],
      [6, 5, 4, 3, 2, 1, 0]
    ]

    in_tensor = ExTorch.tensor(input_list)
    {values, indices} = ExTorch.sort(in_tensor, 1, descending: true)
    assert ExTorch.Tensor.to_list(values) == expected_values
    assert ExTorch.Tensor.to_list(indices) == expected_indices
  end

  test "sort/4" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_values = [
      [10, 5, 4, 3, 0, -1, -2],
      [20, 7, 5, 3, 2, 1, -5],
      [6, 5, 4, 3, 2, 1, 0]
    ]

    expected_indices = [
      [2, 6, 5, 1, 4, 0, 3],
      [5, 4, 0, 3, 2, 6, 1],
      [6, 5, 4, 3, 2, 1, 0]
    ]

    in_tensor = ExTorch.tensor(input_list)
    {values, indices} = ExTorch.sort(in_tensor, 1, true, true)
    assert ExTorch.Tensor.to_list(values) == expected_values
    assert ExTorch.Tensor.to_list(indices) == expected_indices
  end

  test "sort/5" do
    input_list = [
      [-1, 3, 10, -2, 0, 4, 5],
      [5, -5, 2, 3, 7, 20, 1],
      [0, 1, 2, 3, 4, 5, 6]
    ]

    expected_values = [
      [10, 5, 4, 3, 0, -1, -2],
      [20, 7, 5, 3, 2, 1, -5],
      [6, 5, 4, 3, 2, 1, 0]
    ]

    expected_indices = [
      [2, 6, 5, 1, 4, 0, 3],
      [5, 4, 0, 3, 2, 6, 1],
      [6, 5, 4, 3, 2, 1, 0]
    ]

    in_tensor = ExTorch.tensor(input_list, dtype: :float32)
    values = ExTorch.empty_like(in_tensor)
    indices = ExTorch.empty_like(in_tensor, dtype: :int64)
    ExTorch.sort(in_tensor, 1, true, true, {values, indices})
    assert ExTorch.Tensor.to_list(values) == expected_values
    assert ExTorch.Tensor.to_list(indices) == expected_indices
  end

  test "eq/2" do
    expected = ExTorch.tensor([[true, false], [false, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.eq(a, 1)
    assert ExTorch.all(ExTorch.eq(expected, cmp)) |> ExTorch.Tensor.item()
  end

  test "eq/2 with broadcastable" do
    expected = ExTorch.tensor([[true, true], [false, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.eq(a, [1, 2])
    assert ExTorch.all(ExTorch.eq(expected, cmp)) |> ExTorch.Tensor.item()
  end

  test "eq/2 with tensor" do
    expected = ExTorch.tensor([[true, false], [false, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.eq(a, ExTorch.tensor([[1, 1], [4, 4]]))
    assert ExTorch.all(ExTorch.eq(expected, cmp)) |> ExTorch.Tensor.item()
  end

  test "eq/3" do
    a = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.empty_like(a, dtype: :bool)
    expected = ExTorch.tensor([[true, false], [false, true]])
    ExTorch.eq(a, ExTorch.tensor([[1, 1], [4, 4]]), out)
    assert ExTorch.all(ExTorch.eq(expected, out)) |> ExTorch.Tensor.item()
  end

  test "equal/2" do
    assert ExTorch.equal(ExTorch.tensor([1, 2]), ExTorch.tensor([1, 2]))
    assert !ExTorch.equal(ExTorch.tensor([1, 2]), ExTorch.tensor([1]))
  end

  test "ge/2" do
    expected = ExTorch.tensor([[false, true], [true, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.ge(a, 2)
    assert ExTorch.equal(cmp, expected)
  end

  test "ge/2 with broadcastable" do
    expected = ExTorch.tensor([[true, false], [true, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.ge(a, [0, 3])
    assert ExTorch.equal(cmp, expected)
  end

  test "ge/2 with tensor" do
    expected = ExTorch.tensor([[false, true], [true, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.ge(a, ExTorch.tensor([[3, 1], [2, 5]]))
    assert ExTorch.equal(cmp, expected)
  end

  test "ge/3" do
    expected = ExTorch.tensor([[false, true], [true, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.empty_like(a, dtype: :bool)
    ExTorch.ge(a, ExTorch.tensor([[3, 1], [2, 5]]), out)
    assert ExTorch.equal(out, expected)
  end

  test "gt/2" do
    expected = ExTorch.tensor([[false, false], [true, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.gt(a, 2)
    assert ExTorch.equal(cmp, expected)
  end

  test "gt/2 with broadcastable" do
    expected = ExTorch.tensor([[false, false], [true, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.gt(a, [1, 2])
    assert ExTorch.equal(cmp, expected)
  end

  test "gt/2 with tensor" do
    expected = ExTorch.tensor([[false, true], [false, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.gt(a, ExTorch.tensor([[3, 1], [3, 2]]))
    assert ExTorch.equal(cmp, expected)
  end

  test "gt/3" do
    expected = ExTorch.tensor([[false, true], [false, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.empty_like(a, dtype: :bool)
    ExTorch.gt(a, ExTorch.tensor([[3, 1], [3, 2]]), out)
    assert ExTorch.equal(out, expected)
  end

  test "le/2" do
    expected = ExTorch.tensor([[true, true], [false, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.le(a, 2)
    assert ExTorch.equal(cmp, expected)
  end

  test "le/2 with broadcastable" do
    expected = ExTorch.tensor([[true, true], [false, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.le(a, [1, 3])
    assert ExTorch.equal(cmp, expected)
  end

  test "le/2 with tensor" do
    expected = ExTorch.tensor([[true, false], [false, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.le(a, ExTorch.tensor([[3, 1], [2, 5]]))
    assert ExTorch.equal(cmp, expected)
  end

  test "le/3 with tensor" do
    expected = ExTorch.tensor([[true, false], [false, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.empty_like(a, dtype: :bool)
    ExTorch.le(a, ExTorch.tensor([[3, 1], [2, 5]]), out)
    assert ExTorch.equal(out, expected)
  end

  test "lt/2" do
    expected = ExTorch.tensor([[true, false], [false, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.lt(a, 2)
    assert ExTorch.equal(cmp, expected)
  end

  test "lt/2 with broadcastable" do
    expected = ExTorch.tensor([[false, false], [false, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.lt(a, [1, 2])
    assert ExTorch.equal(cmp, expected)
  end

  test "lt/2 with tensor" do
    expected = ExTorch.tensor([[true, false], [false, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.lt(a, ExTorch.tensor([[3, 1], [3, 2]]))
    assert ExTorch.equal(cmp, expected)
  end

  test "lt/3" do
    expected = ExTorch.tensor([[true, false], [false, false]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.empty_like(a, dtype: :bool)
    ExTorch.lt(a, ExTorch.tensor([[3, 1], [3, 2]]), out)
    assert ExTorch.equal(out, expected)
  end

  test "ne/2" do
    expected = ExTorch.tensor([[true, false], [true, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.ne(a, 2)
    assert ExTorch.equal(cmp, expected)
  end

  test "ne/2 with broadcastable" do
    expected = ExTorch.tensor([[false, false], [true, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.ne(a, [1, 2])
    assert ExTorch.equal(cmp, expected)
  end

  test "ne/2 with tensor" do
    expected = ExTorch.tensor([[true, false], [true, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    cmp = ExTorch.ne(a, ExTorch.tensor([[3, 2], [4, -2]]))
    assert ExTorch.equal(cmp, expected)
  end

  test "ne/3" do
    expected = ExTorch.tensor([[true, false], [true, true]])
    a = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.empty_like(a, dtype: :bool)
    ExTorch.ne(a, ExTorch.tensor([[3, 2], [4, -2]]), out)
    assert ExTorch.equal(out, expected)
  end

  test "isclose/2" do
    expected = ExTorch.tensor([true, false])
    t1 = ExTorch.tensor([10_000.0, 1.0e-07])
    t2 = ExTorch.tensor([10_000.1, 1.0e-08])
    assert ExTorch.equal(ExTorch.isclose(t1, t2), expected)
  end

  test "isclose/3" do
    expected = ExTorch.tensor([true, true])
    t1 = ExTorch.tensor([10_000.0, 1.0e-08])
    t2 = ExTorch.tensor([10_000.1, 1.0e-09])
    assert ExTorch.equal(ExTorch.isclose(t1, t2, 1.0e-5), expected)
  end

  test "isclose/3 with kwargs" do
    expected = ExTorch.tensor([true, true, true])
    t1 = ExTorch.tensor([10_000.0, 1.0e-08, :nan])
    t2 = ExTorch.tensor([10_000.1, 1.0e-09, :nan])
    assert ExTorch.equal(ExTorch.isclose(t1, t2, equal_nan: true), expected)
  end

  test "isclose/4" do
    expected = ExTorch.tensor([true, false])
    t1 = ExTorch.tensor([1.0, :nan])
    t2 = ExTorch.tensor([1.0, :nan])
    out = ExTorch.isclose(t1, t2, 1.0e-5, 1.0e-8)
    assert ExTorch.equal(out, expected)
  end

  test "isclose/5" do
    expected = ExTorch.tensor([true, true])
    t1 = ExTorch.tensor([1.0, :nan])
    t2 = ExTorch.tensor([1.0, :nan])
    assert ExTorch.equal(ExTorch.isclose(t1, t2, 1.0e-5, 1.0e-8, true), expected)
  end

  test "isfinite/1" do
    expected = ExTorch.tensor([true, false, true, false, false])
    input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
    isfinite = ExTorch.isfinite(input)
    assert ExTorch.equal(isfinite, expected)
  end

  test "isinf/1" do
    expected = ExTorch.tensor([false, true, false, true, false])
    input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
    isfinite = ExTorch.isinf(input)
    assert ExTorch.equal(isfinite, expected)
  end

  test "isposinf/1" do
    expected = ExTorch.tensor([false, true, false, false, false])
    input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
    isfinite = ExTorch.isposinf(input)
    assert ExTorch.equal(isfinite, expected)
  end

  test "isneginf/1" do
    expected = ExTorch.tensor([false, false, false, true, false])
    input = ExTorch.tensor([1, :inf, 2, :ninf, :nan])
    isfinite = ExTorch.isneginf(input)
    assert ExTorch.equal(isfinite, expected)
  end

  test "isin/2" do
    expected = ExTorch.tensor([[false, true], [false, false]])
    input = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.isin(input, 2)
    assert ExTorch.equal(out, expected)
  end

  test "isin/3" do
    expected = ExTorch.tensor([[true, false], [true, false]])
    input = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.isin(input, [1, 3, 5], true)
    assert ExTorch.equal(out, expected)
  end

  test "isin/3 with kwargs" do
    expected = ExTorch.tensor([[false, true], [false, true]])
    input = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.isin(input, [1, 3, 5], invert: true)
    assert ExTorch.equal(out, expected)
  end

  test "isin/4" do
    expected = ExTorch.tensor([[true, false], [false, true]])
    input = ExTorch.tensor([[1, 2], [3, 4]])
    out = ExTorch.isin(input, ExTorch.tensor([[-1, 3], [2, 5]]), true, true)
    assert ExTorch.equal(out, expected)
  end
end
