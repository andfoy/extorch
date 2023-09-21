defmodule ExTorchTest.Tensor.ReductionTest do
  use ExUnit.Case

  test "all/1" do
    a = ExTorch.full({3, 3}, true)
    assert ExTorch.all(a) |> ExTorch.Tensor.item()

    b = ExTorch.full({3, 3}, false)
    assert !(ExTorch.all(b) |> ExTorch.Tensor.item())
  end

  test "all/1 with kwargs" do
    expected = [false, true, false]
    a = ExTorch.tensor([[true, false, false], [true, true, true], [false, true, true]])
    all_dim = ExTorch.all(a, dim: -1)
    assert ExTorch.Tensor.to_list(all_dim) == expected
  end

  test "all/2" do
    expected = [false, true, false]
    a = ExTorch.tensor([[true, false, false], [true, true, true], [false, true, true]])
    all_dim = ExTorch.all(a, -1)
    assert ExTorch.Tensor.to_list(all_dim) == expected
  end

  test "all/3" do
    expected = [[false], [true], [false]]
    a = ExTorch.tensor([[true, false, false], [true, true, true], [false, true, true]])
    all_dim = ExTorch.all(a, -1, true)
    assert ExTorch.Tensor.to_list(all_dim) == expected
  end

  test "all/3 with kwargs" do
    expected = [[false], [true], [false]]
    a = ExTorch.tensor([[true, false, false], [true, true, true], [false, true, true]])
    all_dim = ExTorch.all(a, -1, keepdim: true)
    assert ExTorch.Tensor.to_list(all_dim) == expected
  end

  test "all/4" do
    expected = [[false], [true], [false]]
    out = ExTorch.empty({3, 1}, dtype: :bool)
    a = ExTorch.tensor([[true, false, false], [true, true, true], [false, true, true]])
    ExTorch.all(a, -1, true, out)
    assert ExTorch.Tensor.to_list(out) == expected
  end

  test "any/1" do
    a = ExTorch.full({3, 3}, true)
    assert ExTorch.any(a) |> ExTorch.Tensor.item()

    b = ExTorch.full({3, 3}, false)
    assert !(ExTorch.any(b) |> ExTorch.Tensor.item())
  end

  test "any/1 with kwargs" do
    expected = [false, true, true]
    a = ExTorch.tensor([[false, false, false], [true, true, true], [false, true, true]])
    any_dim = ExTorch.any(a, dim: -1)
    assert ExTorch.Tensor.to_list(any_dim) == expected
  end

  test "any/2" do
    expected = [false, true, true]
    a = ExTorch.tensor([[false, false, false], [true, true, true], [false, true, true]])
    any_dim = ExTorch.any(a, -1)
    assert ExTorch.Tensor.to_list(any_dim) == expected
  end

  test "any/3" do
    expected = [[false], [true], [true]]
    a = ExTorch.tensor([[false, false, false], [true, true, true], [false, true, true]])
    any_dim = ExTorch.any(a, -1, true)
    assert ExTorch.Tensor.to_list(any_dim) == expected
  end

  test "any/3 with kwargs" do
    expected = [[false], [true], [true]]
    a = ExTorch.tensor([[false, false, false], [true, true, true], [false, true, true]])
    any_dim = ExTorch.any(a, -1, keepdim: true)
    assert ExTorch.Tensor.to_list(any_dim) == expected
  end

  test "any/4" do
    expected = [[false], [true], [true]]
    out = ExTorch.empty({3, 1}, dtype: :bool)
    a = ExTorch.tensor([[false, false, false], [true, true, true], [false, true, true]])
    ExTorch.any(a, -1, true, out)
    assert ExTorch.Tensor.to_list(out) == expected
  end

  test "argmax/1" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    out = ExTorch.argmax(input) |> ExTorch.Tensor.item()
    assert out == 3
  end

  test "argmax/2" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([0, 0, 0])
    out = ExTorch.argmax(input, -1)
    assert ExTorch.equal(out, expected)
  end

  test "argmax/2 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([1, 1, 0])
    out = ExTorch.argmax(input, dim: 0)
    assert ExTorch.equal(out, expected)
  end

  test "argmax/3" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[0], [0], [0]])
    out = ExTorch.argmax(input, -1, true)
    assert ExTorch.equal(out, expected)
  end

  test "argmax/3 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[1, 1, 0]])
    out = ExTorch.argmax(input, 0, keepdim: true)
    assert ExTorch.equal(out, expected)
  end

  test "argmin/1" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    out = ExTorch.argmin(input) |> ExTorch.Tensor.item()
    assert out == 8
  end

  test "argmin/2" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([1, 2, 2])
    out = ExTorch.argmin(input, -1)
    assert ExTorch.equal(out, expected)
  end

  test "argmin/2 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([0, 0, 2])
    out = ExTorch.argmin(input, dim: 0)
    assert ExTorch.equal(out, expected)
  end

  test "argmin/3" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[1], [2], [2]])
    out = ExTorch.argmin(input, -1, true)
    assert ExTorch.equal(out, expected)
  end

  test "argmin/3 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[0, 0, 2]])
    out = ExTorch.argmin(input, 0, keepdim: true)
    assert ExTorch.equal(out, expected)
  end

  test "max/1" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    out = ExTorch.max(input) |> ExTorch.Tensor.item()
    assert out == :inf
  end

  test "max/2" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([3.3, :inf, 10], dtype: :float64)
    expected_argmax = ExTorch.tensor([0, 0, 0], dtype: :long)
    {max, argmax} = ExTorch.max(input, 1)

    assert ExTorch.allclose(max, expected_max)
    assert ExTorch.equal(argmax, expected_argmax)
  end

  test "max/3" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([[:inf, 1, 0.1]], dtype: :float64)
    expected_argmax = ExTorch.tensor([[1, 1, 0]], dtype: :long)
    {max, argmax} = ExTorch.max(input, 0, true)

    assert ExTorch.allclose(max, expected_max)
    assert ExTorch.equal(argmax, expected_argmax)
  end

  test "max/3 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([[:inf, 1, 0.1]], dtype: :float64)
    expected_argmax = ExTorch.tensor([[1, 1, 0]], dtype: :long)
    {max, argmax} = ExTorch.max(input, 0, keepdim: true)

    assert ExTorch.allclose(max, expected_max)
    assert ExTorch.equal(argmax, expected_argmax)
  end

  test "max/4" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    expected_argmax = ExTorch.tensor([[0], [0], [0]], dtype: :long)

    max = ExTorch.empty_like(expected_max)
    argmax = ExTorch.empty_like(expected_argmax)

    ExTorch.max(input, -1, true, {max, argmax})

    assert ExTorch.allclose(max, expected_max)
    assert ExTorch.equal(argmax, expected_argmax)
  end

  test "max/4 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    expected_argmax = ExTorch.tensor([[0], [0], [0]], dtype: :long)

    max = ExTorch.empty_like(expected_max)
    argmax = ExTorch.empty_like(expected_argmax)

    ExTorch.max(input, -1, true, out: {max, argmax})

    assert ExTorch.allclose(max, expected_max)
    assert ExTorch.equal(argmax, expected_argmax)
  end

  test "min/1" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    out = ExTorch.min(input) |> ExTorch.Tensor.item()
    assert out == -10
  end

  test "min/2" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_min = ExTorch.tensor([-2.1, 0.005, -10], dtype: :float64)
    expected_argmin = ExTorch.tensor([1, 2, 2], dtype: :long)
    {min, argmin} = ExTorch.min(input, 1)

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.equal(argmin, expected_argmin)
  end

  test "min/3" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_min = ExTorch.tensor([[3.3, -2.1, -10]], dtype: :float64)
    expected_argmin = ExTorch.tensor([[0, 0, 2]], dtype: :long)
    {min, argmin} = ExTorch.min(input, 0, true)

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.equal(argmin, expected_argmin)
  end

  test "min/3 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_min = ExTorch.tensor([[3.3, -2.1, -10]], dtype: :float64)
    expected_argmin = ExTorch.tensor([[0, 0, 2]], dtype: :long)
    {min, argmin} = ExTorch.min(input, 0, keepdim: true)

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.equal(argmin, expected_argmin)
  end

  test "min/4" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_min = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    expected_argmin = ExTorch.tensor([[1], [2], [2]], dtype: :long)

    min = ExTorch.empty_like(expected_min)
    argmin = ExTorch.empty_like(expected_argmin)

    ExTorch.min(input, -1, true, {min, argmin})

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.equal(argmin, expected_argmin)
  end

  test "min/4 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_min = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    expected_argmin = ExTorch.tensor([[1], [2], [2]], dtype: :long)

    min = ExTorch.empty_like(expected_min)
    argmin = ExTorch.empty_like(expected_argmin)

    ExTorch.min(input, -1, true, out: {min, argmin})

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.equal(argmin, expected_argmin)
  end

  test "amax/2 with integer dim" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([3.3, :inf, 10], dtype: :float64)
    out = ExTorch.amax(input, -1)
    assert ExTorch.allclose(out, expected)
  end

  test "amax/2 with tuple dims" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    out = ExTorch.amax(input, {0, 1}) |> ExTorch.Tensor.item()
    assert out == :inf
  end

  test "amax/3" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    out = ExTorch.amax(input, -1, true)
    assert ExTorch.allclose(out, expected)
  end

  test "amax/3 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    out = ExTorch.amax(input, -1, keepdim: true)
    assert ExTorch.allclose(out, expected)
  end

  test "amax/4" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    out = ExTorch.empty_like(expected)

    ExTorch.amax(input, -1, true, out)
    assert ExTorch.allclose(out, expected)
  end

  test "amin/2 with integer dim" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([-2.1, 0.005, -10], dtype: :float64)
    out = ExTorch.amin(input, -1)
    assert ExTorch.allclose(out, expected)
  end

  test "amin/2 with tuple dims" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    out = ExTorch.amin(input, {0, 1}) |> ExTorch.Tensor.item()
    assert out == -10
  end

  test "amin/3" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    out = ExTorch.amin(input, -1, true)
    assert ExTorch.allclose(out, expected)
  end

  test "amin/3 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    out = ExTorch.amin(input, -1, keepdim: true)
    assert ExTorch.allclose(out, expected)
  end

  test "amin/4" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    out = ExTorch.empty_like(expected)

    ExTorch.amin(input, -1, true, out)
    assert ExTorch.allclose(out, expected)
  end

  test "aminmax/1" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    {min, max} = ExTorch.aminmax(input)
    assert ExTorch.Tensor.item(min) == -10
    assert ExTorch.Tensor.item(max) == :inf
  end

  test "aminmax/2" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([3.3, :inf, 10], dtype: :float64)
    expected_min = ExTorch.tensor([-2.1, 0.005, -10], dtype: :float64)
    {min, max} = ExTorch.aminmax(input, -1)

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.allclose(max, expected_max)
  end

  test "aminmax/2 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([3.3, :inf, 10], dtype: :float64)
    expected_min = ExTorch.tensor([-2.1, 0.005, -10], dtype: :float64)
    {min, max} = ExTorch.aminmax(input, dim: -1)

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.allclose(max, expected_max)
  end

  test "aminmax/3" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    expected_min = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    {min, max} = ExTorch.aminmax(input, -1, true)

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.allclose(max, expected_max)
  end

  test "aminmax/3 with kwargs" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    expected_min = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    {min, max} = ExTorch.aminmax(input, -1, keepdim: true)

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.allclose(max, expected_max)
  end

  test "aminmax/4" do
    input = ExTorch.tensor([
      [3.3, -2.1, 0.1],
      [:inf, 1, 0.005],
      [10, 0.5, -10]
    ])

    expected_max = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    expected_min = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)

    max = ExTorch.empty_like(expected_max)
    min = ExTorch.empty_like(expected_min)
    ExTorch.aminmax(input, -1, true, {min, max})

    assert ExTorch.allclose(min, expected_min)
    assert ExTorch.allclose(max, expected_max)
  end

  test "dist/2" do
    a = ExTorch.tensor([1.0, 2, -3])
    b = ExTorch.tensor([0.0, -1, 4])
    expected = :math.sqrt(1 + 9 + 49)
    out = ExTorch.dist(a, b) |> ExTorch.Tensor.item()
    assert expected == out
  end

  test "dist/3" do
    a = ExTorch.tensor([1.0, 2, -3])
    b = ExTorch.tensor([0.0, -1, 4])
    expected = 1 + 3 + 7.0
    out = ExTorch.dist(a, b, 1) |> ExTorch.Tensor.item()
    assert expected == out
  end

  test "dist/3 with kwargs" do
    a = ExTorch.tensor([1.0, 2, -3])
    b = ExTorch.tensor([0.0, -1, 4])
    expected = 7.0
    out = ExTorch.dist(a, b, p: :inf) |> ExTorch.Tensor.item()
    assert expected == out
  end

end
