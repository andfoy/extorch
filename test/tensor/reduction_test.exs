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
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    out = ExTorch.argmax(input) |> ExTorch.Tensor.item()
    assert out == 3
  end

  test "argmax/2" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([0, 0, 0])
    out = ExTorch.argmax(input, -1)
    assert ExTorch.equal(out, expected)
  end

  test "argmax/2 with kwargs" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([1, 1, 0])
    out = ExTorch.argmax(input, dim: 0)
    assert ExTorch.equal(out, expected)
  end

  test "argmax/3" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([[0], [0], [0]])
    out = ExTorch.argmax(input, -1, true)
    assert ExTorch.equal(out, expected)
  end

  test "argmax/3 with kwargs" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([[1, 1, 0]])
    out = ExTorch.argmax(input, 0, keepdim: true)
    assert ExTorch.equal(out, expected)
  end

  test "argmin/1" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    out = ExTorch.argmin(input) |> ExTorch.Tensor.item()
    assert out == 8
  end

  test "argmin/2" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([1, 2, 2])
    out = ExTorch.argmin(input, -1)
    assert ExTorch.equal(out, expected)
  end

  test "argmin/2 with kwargs" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([0, 0, 2])
    out = ExTorch.argmin(input, dim: 0)
    assert ExTorch.equal(out, expected)
  end

  test "argmin/3" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([[1], [2], [2]])
    out = ExTorch.argmin(input, -1, true)
    assert ExTorch.equal(out, expected)
  end

  test "argmin/3 with kwargs" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([[0, 0, 2]])
    out = ExTorch.argmin(input, 0, keepdim: true)
    assert ExTorch.equal(out, expected)
  end

  test "max/1" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    out = ExTorch.max(input) |> ExTorch.Tensor.item()
    assert out == :inf
  end

  test "max/2" do
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    out = ExTorch.min(input) |> ExTorch.Tensor.item()
    assert out == -10
  end

  test "min/2" do
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([3.3, :inf, 10], dtype: :float64)
    out = ExTorch.amax(input, -1)
    assert ExTorch.allclose(out, expected)
  end

  test "amax/2 with tuple dims" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    out = ExTorch.amax(input, {0, 1}) |> ExTorch.Tensor.item()
    assert out == :inf
  end

  test "amax/3" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    out = ExTorch.amax(input, -1, true)
    assert ExTorch.allclose(out, expected)
  end

  test "amax/3 with kwargs" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([[3.3], [:inf], [10]], dtype: :float64)
    out = ExTorch.amax(input, -1, keepdim: true)
    assert ExTorch.allclose(out, expected)
  end

  test "amax/4" do
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([-2.1, 0.005, -10], dtype: :float64)
    out = ExTorch.amin(input, -1)
    assert ExTorch.allclose(out, expected)
  end

  test "amin/2 with tuple dims" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    out = ExTorch.amin(input, {0, 1}) |> ExTorch.Tensor.item()
    assert out == -10
  end

  test "amin/3" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    out = ExTorch.amin(input, -1, true)
    assert ExTorch.allclose(out, expected)
  end

  test "amin/3 with kwargs" do
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    expected = ExTorch.tensor([[-2.1], [0.005], [-10]], dtype: :float64)
    out = ExTorch.amin(input, -1, keepdim: true)
    assert ExTorch.allclose(out, expected)
  end

  test "amin/4" do
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
        [3.3, -2.1, 0.1],
        [:inf, 1, 0.005],
        [10, 0.5, -10]
      ])

    {min, max} = ExTorch.aminmax(input)
    assert ExTorch.Tensor.item(min) == -10
    assert ExTorch.Tensor.item(max) == :inf
  end

  test "aminmax/2" do
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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
    input =
      ExTorch.tensor([
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

  test "logsumexp/1" do
    a =
      ExTorch.tensor([
        [0.2292, -1.0899, 0.0889],
        [-2.0117, 0.4716, -0.3893],
        [-0.9382, 1.0590, -0.0838]
      ])

    expected = ExTorch.tensor(2.2295)
    out = ExTorch.logsumexp(a)

    assert ExTorch.allclose(expected, out)
  end

  test "logsumexp/2" do
    a =
      ExTorch.tensor([
        [0.2292, -1.0899, 0.0889],
        [-2.0117, 0.4716, -0.3893],
        [-0.9382, 1.0590, -0.0838]
      ])

    expected = ExTorch.tensor([0.9883, 0.8812, 1.4338])
    out = ExTorch.logsumexp(a, -1)

    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "logsumexp/2 with kwargs" do
    a =
      ExTorch.tensor([
        [0.2292, -1.0899, 0.0889],
        [-2.0117, 0.4716, -0.3893],
        [-0.9382, 1.0590, -0.0838]
      ])

    expected = ExTorch.tensor([0.9883, 0.8812, 1.4338])
    out = ExTorch.logsumexp(a, dim: -1)

    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "logsumexp/3" do
    a =
      ExTorch.tensor([
        [0.2292, -1.0899, 0.0889],
        [-2.0117, 0.4716, -0.3893],
        [-0.9382, 1.0590, -0.0838]
      ])

    expected = ExTorch.tensor([[0.9883], [0.8812], [1.4338]])
    out = ExTorch.logsumexp(a, -1, true)

    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "logsumexp/3 with kwargs" do
    a =
      ExTorch.tensor([
        [0.2292, -1.0899, 0.0889],
        [-2.0117, 0.4716, -0.3893],
        [-0.9382, 1.0590, -0.0838]
      ])

    expected = ExTorch.tensor([[0.9883], [0.8812], [1.4338]])
    out = ExTorch.logsumexp(a, -1, keepdim: true)

    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "logsumexp/4" do
    a =
      ExTorch.tensor([
        [0.2292, -1.0899, 0.0889],
        [-2.0117, 0.4716, -0.3893],
        [-0.9382, 1.0590, -0.0838]
      ])

    expected = ExTorch.tensor([[0.9883], [0.8812], [1.4338]])
    out = ExTorch.empty_like(expected)
    out = ExTorch.logsumexp(a, -1, true, out)

    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "logsumexp/4 with kwargs" do
    a =
      ExTorch.tensor([
        [0.2292, -1.0899, 0.0889],
        [-2.0117, 0.4716, -0.3893],
        [-0.9382, 1.0590, -0.0838]
      ])

    expected = ExTorch.tensor([[0.9883], [0.8812], [1.4338]])
    out = ExTorch.empty_like(expected)
    out = ExTorch.logsumexp(a, -1, true, out: out)

    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "sum/1" do
    a =
      ExTorch.tensor([
        [0.7281, 0.9280, 0.5829],
        [0.4569, 0.4785, 0.1352],
        [0.9905, 0.0698, 0.1905]
      ])

    expected = ExTorch.tensor(4.5604)
    out = ExTorch.sum(a)
    assert ExTorch.allclose(expected, out)
  end

  test "sum/2" do
    a =
      ExTorch.tensor([
        [0.7281, 0.9280, 0.5829],
        [0.4569, 0.4785, 0.1352],
        [0.9905, 0.0698, 0.1905]
      ])

    expected = ExTorch.tensor([2.1755, 1.4764, 0.9086])
    out = ExTorch.sum(a, 0)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "sum/2 with kwargs" do
    a =
      ExTorch.tensor([
        [0.7281, 0.9280, 0.5829],
        [0.4569, 0.4785, 0.1352],
        [0.9905, 0.0698, 0.1905]
      ])

    expected = ExTorch.tensor(4.5604, dtype: :double)
    out = ExTorch.sum(a, dtype: :double)
    assert out.dtype == :double
    assert ExTorch.allclose(expected, out)
  end

  test "sum/3" do
    a =
      ExTorch.tensor([
        [0.7281, 0.9280, 0.5829],
        [0.4569, 0.4785, 0.1352],
        [0.9905, 0.0698, 0.1905]
      ])

    expected = ExTorch.tensor([[2.1755, 1.4764, 0.9086]])
    out = ExTorch.sum(a, 0, true)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "sum/3 with kwargs" do
    a =
      ExTorch.tensor([
        [0.7281, 0.9280, 0.5829],
        [0.4569, 0.4785, 0.1352],
        [0.9905, 0.0698, 0.1905]
      ])

    expected = ExTorch.tensor([[2.1755, 1.4764, 0.9086]])
    out = ExTorch.sum(a, 0, keepdim: true)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "sum/4" do
    a =
      ExTorch.tensor([
        [0.7281, 0.9280, 0.5829],
        [0.4569, 0.4785, 0.1352],
        [0.9905, 0.0698, 0.1905]
      ])

    expected = ExTorch.tensor([[2.1755, 1.4764, 0.9086]], dtype: :double)
    out = ExTorch.sum(a, 0, true, :double)
    assert out.dtype == :double
    assert ExTorch.allclose(expected, out, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "mean/1" do
    input =
      ExTorch.tensor([
        [0.0945, 0.3992, 0.5090],
        [0.0142, 0.1471, 0.4568],
        [0.1428, 0.2121, 0.6163]
      ])

    expected = ExTorch.tensor(0.2880)
    out = ExTorch.mean(input)
    assert ExTorch.allclose(expected, out)
  end

  test "mean/2" do
    input =
      ExTorch.tensor([
        [0.0945, 0.3992, 0.5090],
        [0.0142, 0.1471, 0.4568],
        [0.1428, 0.2121, 0.6163]
      ])

    expected = ExTorch.tensor([0.0838, 0.2528, 0.5274])
    out = ExTorch.mean(input, 0)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "mean/2 with kwargs" do
    input =
      ExTorch.tensor([
        [0.0945, 0.3992, 0.5090],
        [0.0142, 0.1471, 0.4568],
        [0.1428, 0.2121, 0.6163]
      ])

    expected = ExTorch.tensor(0.2880, dtype: :double)
    out = ExTorch.mean(input, dtype: :double)
    assert out.dtype == :double
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "mean/3" do
    input =
      ExTorch.tensor([
        [0.0945, 0.3992, 0.5090],
        [0.0142, 0.1471, 0.4568],
        [0.1428, 0.2121, 0.6163]
      ])

    expected = ExTorch.tensor([[0.3342], [0.2060], [0.3237]])
    out = ExTorch.mean(input, -1, true)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "mean/3 with kwargs" do
    input =
      ExTorch.tensor([
        [0.0945, 0.3992, 0.5090],
        [0.0142, 0.1471, 0.4568],
        [0.1428, 0.2121, 0.6163]
      ])

    expected = ExTorch.tensor([[0.3342], [0.2060], [0.3237]])
    out = ExTorch.mean(input, -1, keepdim: true)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "mean/4" do
    input =
      ExTorch.tensor([
        [0.0945, 0.3992, 0.5090],
        [0.0142, 0.1471, 0.4568],
        [0.1428, 0.2121, 0.6163]
      ])

    expected = ExTorch.tensor([[0.3342], [0.2060], [0.3237]], dtype: :float64)
    out = ExTorch.mean(input, -1, true, :double)
    assert out.dtype == :double
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "mean/4 with kwargs" do
    input =
      ExTorch.tensor([
        [0.0945, 0.3992, 0.5090],
        [0.0142, 0.1471, 0.4568],
        [0.1428, 0.2121, 0.6163]
      ])

    expected = ExTorch.tensor([[0.3342], [0.2060], [0.3237]], dtype: :float64)
    out = ExTorch.mean(input, -1, true, dtype: :double)
    assert out.dtype == :double
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "mean/5" do
    input =
      ExTorch.tensor([
        [0.0945, 0.3992, 0.5090],
        [0.0142, 0.1471, 0.4568],
        [0.1428, 0.2121, 0.6163]
      ])

    expected = ExTorch.tensor([[0.3342], [0.2060], [0.3237]])
    out = ExTorch.empty_like(expected)
    ExTorch.mean(input, -1, true, nil, out)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "nanmean/1" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected = ExTorch.tensor(0.6667)
    out = ExTorch.nanmean(input)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "nanmean/2" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected =
      ExTorch.tensor([1.5000, 0.5000, 0.0000])

    out = ExTorch.nanmean(input, -1)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "nanmean/2 with kwargs" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected =
      ExTorch.tensor([
        [1.5000],
        [0.5000],
        [0.0000]
      ])

    out = ExTorch.nanmean(input, dim: -1, keepdim: true)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "nanmean/3" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected =
      ExTorch.tensor([
        [1.5000],
        [0.5000],
        [0.0000]
      ])

    out = ExTorch.nanmean(input, -1, true)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "nanmean/3 with kwargs" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected =
      ExTorch.tensor([
        [1.5000],
        [0.5000],
        [0.0000]
      ])

    out = ExTorch.nanmean(input, -1, keepdim: true)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "nanmean/4" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected =
      ExTorch.tensor(
        [
          [1.5000],
          [0.5000],
          [0.0000]
        ],
        dtype: :double
      )

    out = ExTorch.nanmean(input, -1, true, :double)
    assert out.dtype == :double
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "nanmean/4 with kwargs" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected =
      ExTorch.tensor([
        [1.5000],
        [0.5000],
        [0.0000]
      ])

    out = ExTorch.nanmean(input, -1, true, dtype: nil)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "nanmean/5" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected =
      ExTorch.tensor([
        [1.5000],
        [0.5000],
        [0.0000]
      ])

    out = ExTorch.empty_like(expected)
    ExTorch.nanmean(input, -1, true, nil, out)
    assert ExTorch.allclose(expected, out, rtol: 1.0e-4, atol: 1.0e-4)
  end

  test "median/1" do
    input =
      ExTorch.tensor([
        [-0.7721, -2.0910, -0.4622],
        [0.1119, 2.4266, 1.3471],
        [-0.1450, -0.2876, -2.3025]
      ])

    expected = ExTorch.tensor(-0.2876)
    out = ExTorch.median(input)
    assert ExTorch.allclose(expected, out)
  end

  test "median/2" do
    input =
      ExTorch.tensor([
        [-0.7721, -2.0910, -0.4622],
        [0.1119, 2.4266, 1.3471],
        [-0.1450, -0.2876, -2.3025]
      ])

    expected_values = ExTorch.tensor([-0.7721, 1.3471, -0.2876])
    expected_indices = ExTorch.tensor([0, 2, 1], dtype: :long)
    {values, indices} = ExTorch.median(input, -1)

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "median/2 with kwargs" do
    input =
      ExTorch.tensor([
        [-0.7721, -2.0910, -0.4622],
        [0.1119, 2.4266, 1.3471],
        [-0.1450, -0.2876, -2.3025]
      ])

    expected_values = ExTorch.tensor([-0.7721, 1.3471, -0.2876])
    expected_indices = ExTorch.tensor([0, 2, 1], dtype: :long)
    {values, indices} = ExTorch.median(input, dim: -1)

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "median/3" do
    input =
      ExTorch.tensor([
        [-0.7721, -2.0910, -0.4622],
        [0.1119, 2.4266, 1.3471],
        [-0.1450, -0.2876, -2.3025]
      ])

    expected_values =
      ExTorch.tensor([
        [-0.7721],
        [1.3471],
        [-0.2876]
      ])

    expected_indices =
      ExTorch.tensor(
        [
          [0],
          [2],
          [1]
        ],
        dtype: :long
      )

    {values, indices} = ExTorch.median(input, -1, true)

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "median/3 with kwargs" do
    input =
      ExTorch.tensor([
        [-0.7721, -2.0910, -0.4622],
        [0.1119, 2.4266, 1.3471],
        [-0.1450, -0.2876, -2.3025]
      ])

    expected_values =
      ExTorch.tensor([
        [-0.7721],
        [1.3471],
        [-0.2876]
      ])

    expected_indices =
      ExTorch.tensor(
        [
          [0],
          [2],
          [1]
        ],
        dtype: :long
      )

    {values, indices} = ExTorch.median(input, -1, keepdim: true)

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "median/4" do
    input =
      ExTorch.tensor([
        [-0.7721, -2.0910, -0.4622],
        [0.1119, 2.4266, 1.3471],
        [-0.1450, -0.2876, -2.3025]
      ])

    expected_values =
      ExTorch.tensor([
        [-0.7721],
        [1.3471],
        [-0.2876]
      ])

    expected_indices =
      ExTorch.tensor(
        [
          [0],
          [2],
          [1]
        ],
        dtype: :long
      )

    values = ExTorch.empty_like(expected_values)
    indices = ExTorch.empty_like(expected_indices)
    ExTorch.median(input, -1, true, {values, indices})

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "nanmedian/1" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected = ExTorch.tensor(1.0)
    out = ExTorch.nanmedian(input)
    assert ExTorch.allclose(expected, out)
  end

  test "nanmedian/2" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected_values = ExTorch.tensor([1.0, -1.0, -1.0])
    expected_indices = ExTorch.tensor([1, 0, 1], dtype: :long)
    {values, indices} = ExTorch.nanmedian(input, -1)

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "nanmedian/2 with kwargs" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected_values = ExTorch.tensor([1.0, -1.0, -1.0])
    expected_indices = ExTorch.tensor([1, 0, 1], dtype: :long)
    {values, indices} = ExTorch.nanmedian(input, dim: -1)

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "nanmedian/3" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected_values =
      ExTorch.tensor([
        [1.0],
        [-1.0],
        [-1.0]
      ])

    expected_indices =
      ExTorch.tensor(
        [
          [1],
          [0],
          [1]
        ],
        dtype: :long
      )

    {values, indices} = ExTorch.nanmedian(input, -1, true)

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "nanmedian/3 with kwargs" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected_values =
      ExTorch.tensor([
        [1.0],
        [-1.0],
        [-1.0]
      ])

    expected_indices =
      ExTorch.tensor(
        [
          [1],
          [0],
          [1]
        ],
        dtype: :long
      )

    {values, indices} = ExTorch.nanmedian(input, -1, keepdim: true)

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "nanmedian/4" do
    input =
      ExTorch.tensor([
        [:nan, 1.0, 2.0],
        [-1.0, :nan, 2.0],
        [1.0, -1.0, :nan]
      ])

    expected_values =
      ExTorch.tensor([
        [1.0],
        [-1.0],
        [-1.0]
      ])

    expected_indices =
      ExTorch.tensor(
        [
          [1],
          [0],
          [1]
        ],
        dtype: :long
      )

    values = ExTorch.empty_like(expected_values)
    indices = ExTorch.empty_like(expected_indices)
    ExTorch.nanmedian(input, -1, true, {values, indices})

    assert ExTorch.allclose(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "mode/1" do
    input =
      ExTorch.tensor(
        [
          [4, 4, 4, 4],
          [3, 4, 1, 1],
          [3, 2, 2, 0]
        ],
        dtype: :int32
      )

    expected_values = ExTorch.tensor([4, 1, 2], dtype: :int32)
    expected_indices = ExTorch.tensor([3, 3, 2], dtype: :int64)

    {values, indices} = ExTorch.mode(input)

    assert ExTorch.equal(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "mode/2" do
    input =
      ExTorch.tensor(
        [
          [4, 4, 4, 4],
          [3, 4, 1, 1],
          [3, 2, 2, 0]
        ],
        dtype: :int32
      )

    expected_values = ExTorch.tensor([3, 4, 1, 0], dtype: :int32)
    expected_indices = ExTorch.tensor([2, 1, 1, 2], dtype: :int64)

    {values, indices} = ExTorch.mode(input, 0)

    assert ExTorch.equal(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "mode/2 with kwargs" do
    input =
      ExTorch.tensor(
        [
          [4, 4, 4, 4],
          [3, 4, 1, 1],
          [3, 2, 2, 0]
        ],
        dtype: :int32
      )

    expected_values = ExTorch.tensor([3, 4, 1, 0], dtype: :int32)
    expected_indices = ExTorch.tensor([2, 1, 1, 2], dtype: :int64)

    {values, indices} = ExTorch.mode(input, dim: 0)

    assert ExTorch.equal(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "mode/3" do
    input =
      ExTorch.tensor(
        [
          [4, 4, 4, 4],
          [3, 4, 1, 1],
          [3, 2, 2, 0]
        ],
        dtype: :int32
      )

    expected_values =
      ExTorch.tensor(
        [
          [4],
          [1],
          [2]
        ],
        dtype: :int32
      )

    expected_indices =
      ExTorch.tensor(
        [
          [3],
          [3],
          [2]
        ],
        dtype: :int64
      )

    {values, indices} = ExTorch.mode(input, -1, true)

    assert ExTorch.equal(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "mode/3 with kwargs" do
    input =
      ExTorch.tensor(
        [
          [4, 4, 4, 4],
          [3, 4, 1, 1],
          [3, 2, 2, 0]
        ],
        dtype: :int32
      )

    expected_values =
      ExTorch.tensor(
        [
          [4],
          [1],
          [2]
        ],
        dtype: :int32
      )

    expected_indices =
      ExTorch.tensor(
        [
          [3],
          [3],
          [2]
        ],
        dtype: :int64
      )

    {values, indices} = ExTorch.mode(input, -1, keepdim: true)

    assert ExTorch.equal(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "mode/4" do
    input =
      ExTorch.tensor(
        [
          [4, 4, 4, 4],
          [3, 4, 1, 1],
          [3, 2, 2, 0]
        ],
        dtype: :int32
      )

    expected_values =
      ExTorch.tensor(
        [
          [4],
          [1],
          [2]
        ],
        dtype: :int32
      )

    expected_indices =
      ExTorch.tensor(
        [
          [3],
          [3],
          [2]
        ],
        dtype: :int64
      )

    values = ExTorch.empty_like(expected_values)
    indices = ExTorch.empty_like(expected_indices)
    ExTorch.mode(input, -1, true, {values, indices})

    assert ExTorch.equal(expected_values, values)
    assert ExTorch.equal(expected_indices, indices)
  end

  test "nansum/1" do
    input =
      ExTorch.tensor([
        [4, 4, 4, :nan],
        [3, :nan, 1, 1],
        [3, 2, :nan, 0]
      ])

    expected = ExTorch.tensor(22.0)
    out = ExTorch.nansum(input)
    assert ExTorch.allclose(expected, out)
  end

  test "nansum/2" do
    input =
      ExTorch.tensor([
        [4, 4, 4, :nan],
        [3, :nan, 1, 1],
        [3, 2, :nan, 0]
      ])

    expected = ExTorch.tensor([12.0, 5.0, 5.0])
    out = ExTorch.nansum(input, -1)
    assert ExTorch.allclose(expected, out)
  end

  test "nansum/2 with kwargs" do
    input =
      ExTorch.tensor([
        [4, 4, 4, :nan],
        [3, :nan, 1, 1],
        [3, 2, :nan, 0]
      ])

    expected = ExTorch.tensor([12.0, 5.0, 5.0])
    out = ExTorch.nansum(input, dim: -1)
    assert ExTorch.allclose(expected, out)
  end

  test "nansum/3" do
    input =
      ExTorch.tensor([
        [4, 4, 4, :nan],
        [3, :nan, 1, 1],
        [3, 2, :nan, 0]
      ])

    expected =
      ExTorch.tensor([
        [12.0],
        [5.0],
        [5.0]
      ])

    out = ExTorch.nansum(input, -1, true)
    assert ExTorch.allclose(expected, out)
  end

  test "nansum/3 with kwargs" do
    input =
      ExTorch.tensor([
        [4, 4, 4, :nan],
        [3, :nan, 1, 1],
        [3, 2, :nan, 0]
      ])

    expected =
      ExTorch.tensor([
        [12.0],
        [5.0],
        [5.0]
      ])

    out = ExTorch.nansum(input, -1, keepdim: true)
    assert ExTorch.allclose(expected, out)
  end

  test "nansum/4" do
    input =
      ExTorch.tensor([
        [4, 4, 4, :nan],
        [3, :nan, 1, 1],
        [3, 2, :nan, 0]
      ])

    expected =
      ExTorch.tensor(
        [
          [12.0],
          [5.0],
          [5.0]
        ],
        dtype: :double
      )

    out = ExTorch.nansum(input, -1, true, :double)
    assert out.dtype == :double
    assert ExTorch.allclose(expected, out)
  end

  test "prod/1" do
    input =
      ExTorch.tensor([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 1.0]
      ])

    expected = ExTorch.tensor(64.0)
    out = ExTorch.prod(input)
    assert ExTorch.allclose(expected, out)
  end

  test "prod/2" do
    input =
      ExTorch.tensor([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 1.0]
      ])

    expected = ExTorch.tensor([2.0, 4.0, 8.0])
    out = ExTorch.prod(input, -1)
    assert ExTorch.allclose(expected, out)
  end

  test "prod/2 with kwargs" do
    input =
      ExTorch.tensor([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 1.0]
      ])

    expected = ExTorch.tensor([2.0, 4.0, 8.0])
    out = ExTorch.prod(input, dim: -1)
    assert ExTorch.allclose(expected, out)
  end

  test "prod/3" do
    input =
      ExTorch.tensor([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 1.0]
      ])

    expected =
      ExTorch.tensor([
        [2.0],
        [4.0],
        [8.0]
      ])

    out = ExTorch.prod(input, -1, true)
    assert ExTorch.allclose(expected, out)
  end

  test "prod/3 with kwargs" do
    input =
      ExTorch.tensor([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 1.0]
      ])

    expected =
      ExTorch.tensor([
        [2.0],
        [4.0],
        [8.0]
      ])

    out = ExTorch.prod(input, -1, keepdim: true)
    assert ExTorch.allclose(expected, out)
  end

  test "prod/4" do
    input =
      ExTorch.tensor([
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 1.0]
      ])

    expected =
      ExTorch.tensor(
        [
          [2.0],
          [4.0],
          [8.0]
        ],
        dtype: :double
      )

    out = ExTorch.prod(input, -1, true, :double)
    assert out.dtype == :double
    assert ExTorch.allclose(expected, out)
  end

  test "quantile/2" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])
    expected = ExTorch.tensor([1.2500, 2.5000, 3.7500])

    out = ExTorch.quantile(input, q)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/3" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [0.5000, 3.5000],
        [1.0000, 4.0000],
        [1.5000, 4.5000]
      ])

    out = ExTorch.quantile(input, q, -1)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/3 with kwargs" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [0.5000, 3.5000],
        [1.0000, 4.0000],
        [1.5000, 4.5000]
      ])

    out = ExTorch.quantile(input, q, dim: -1)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/4" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [[0.5000], [3.5000]],
        [[1.0000], [4.0000]],
        [[1.5000], [4.5000]]
      ])

    out = ExTorch.quantile(input, q, -1, true)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/4 with kwargs" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [[0.5000], [3.5000]],
        [[1.0000], [4.0000]],
        [[1.5000], [4.5000]]
      ])

    out = ExTorch.quantile(input, q, -1, keepdim: true)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/5 (linear interpolate)" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([1.2500, 2.5000, 3.7500])

    out = ExTorch.quantile(input, q, nil, false, :linear)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/5 (lower interpolate)" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([1.0, 2.0, 3.0])

    out = ExTorch.quantile(input, q, nil, false, :lower)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/5 (higher interpolate)" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([2.0, 3.0, 4.0])

    out = ExTorch.quantile(input, q, nil, false, :higher)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/5 (midpoint interpolate)" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([1.5000, 2.5000, 3.5000])

    out = ExTorch.quantile(input, q, nil, false, :midpoint)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/5 (nearest interpolate)" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([1.0, 2.0, 4.0])

    out = ExTorch.quantile(input, q, nil, false, :nearest)
    assert ExTorch.allclose(out, expected)
  end

  test "quantile/6" do
    input = ExTorch.arange(6) |> ExTorch.reshape({2, 3})
    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [[0.5000], [3.5000]],
        [[1.0000], [4.0000]],
        [[1.5000], [4.5000]]
      ])

    out = ExTorch.empty_like(expected)
    ExTorch.quantile(input, q, -1, true, :linear, out)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/2" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])
    expected = ExTorch.tensor([2.0, 3.0, 4.0])

    out = ExTorch.nanquantile(input, q)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/3" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [0.5000, 3.5000],
        [1.0000, 4.0000],
        [1.5000, 4.5000]
      ])

    out = ExTorch.nanquantile(input, q, -1)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/3 with kwargs" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [0.5000, 3.5000],
        [1.0000, 4.0000],
        [1.5000, 4.5000]
      ])

    out = ExTorch.nanquantile(input, q, dim: -1)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/4" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [[0.5000], [3.5000]],
        [[1.0000], [4.0000]],
        [[1.5000], [4.5000]]
      ])

    out = ExTorch.nanquantile(input, q, -1, true)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/4 with kwargs" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [[0.5000], [3.5000]],
        [[1.0000], [4.0000]],
        [[1.5000], [4.5000]]
      ])

    out = ExTorch.nanquantile(input, q, -1, keepdim: true)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/5 (linear interpolate)" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])
    expected = ExTorch.tensor([2.0, 3.0, 4.0])

    out = ExTorch.nanquantile(input, q, nil, false, :linear)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/5 (lower interpolate)" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([2.0, 3.0, 4.0])

    out = ExTorch.nanquantile(input, q, nil, false, :lower)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/5 (higher interpolate)" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([2.0, 3.0, 4.0])

    out = ExTorch.nanquantile(input, q, nil, false, :higher)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/5 (midpoint interpolate)" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([2.0, 3.0, 4.0])

    out = ExTorch.nanquantile(input, q, nil, false, :midpoint)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/5 (nearest interpolate)" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected = ExTorch.tensor([2.0, 3.0, 4.0])

    out = ExTorch.nanquantile(input, q, nil, false, :nearest)
    assert ExTorch.allclose(out, expected)
  end

  test "nanquantile/6" do
    input =
      ExTorch.arange(6)
      |> ExTorch.reshape({2, 3})
      |> ExTorch.index_put({0, 1}, :nan)

    q = ExTorch.tensor([0.25, 0.5, 0.75])

    expected =
      ExTorch.tensor([
        [[0.5000], [3.5000]],
        [[1.0000], [4.0000]],
        [[1.5000], [4.5000]]
      ])

    out = ExTorch.empty_like(expected)
    ExTorch.nanquantile(input, q, -1, true, :linear, out)
    assert ExTorch.allclose(out, expected)
  end

  test "std/1" do
    input =
      ExTorch.tensor([
        [0.0686, 0.7169, 0.2143, 1.5755],
        [-1.6080, 0.9169, -0.0937, 1.2906],
        [0.5432, 2.4151, -0.3814, 0.2830],
        [-0.0724, 0.7037, -0.1951, -0.1191]
      ])

    expected = ExTorch.tensor(0.9167)
    out = ExTorch.std(input)
    assert ExTorch.allclose(out, expected, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "std/2" do
    input =
      ExTorch.tensor([
        [0.0686, 0.7169, 0.2143, 1.5755],
        [-1.6080, 0.9169, -0.0937, 1.2906],
        [0.5432, 2.4151, -0.3814, 0.2830],
        [-0.0724, 0.7037, -0.1951, -0.1191]
      ])

    expected = ExTorch.tensor([0.6804, 1.2957, 1.1984, 0.4193])
    out = ExTorch.std(input, -1)
    assert ExTorch.allclose(out, expected, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "std/2 with kwargs" do
    input =
      ExTorch.tensor([
        [0.0686, 0.7169, 0.2143, 1.5755],
        [-1.6080, 0.9169, -0.0937, 1.2906],
        [0.5432, 2.4151, -0.3814, 0.2830],
        [-0.0724, 0.7037, -0.1951, -0.1191]
      ])

    expected = ExTorch.tensor([0.6804, 1.2957, 1.1984, 0.4193])
    out = ExTorch.std(input, dim: 1)
    assert ExTorch.allclose(out, expected, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "std/3" do
    input =
      ExTorch.tensor([
        [0.0686, 0.7169, 0.2143, 1.5755],
        [-1.6080, 0.9169, -0.0937, 1.2906],
        [0.5432, 2.4151, -0.3814, 0.2830],
        [-0.0724, 0.7037, -0.1951, -0.1191]
      ])

    expected = ExTorch.tensor([0.8333, 1.5870, 1.4677, 0.5136])
    out = ExTorch.std(input, -1, 2)
    assert ExTorch.allclose(out, expected, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "std/3 with kwargs" do
    input =
      ExTorch.tensor([
        [0.0686, 0.7169, 0.2143, 1.5755],
        [-1.6080, 0.9169, -0.0937, 1.2906],
        [0.5432, 2.4151, -0.3814, 0.2830],
        [-0.0724, 0.7037, -0.1951, -0.1191]
      ])

    expected =
      ExTorch.tensor([
        [0.6804],
        [1.2957],
        [1.1984],
        [0.4193]
      ])

    out = ExTorch.std(input, -1, keepdim: true)
    assert ExTorch.allclose(out, expected, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "std/4" do
    input =
      ExTorch.tensor([
        [0.0686, 0.7169, 0.2143, 1.5755],
        [-1.6080, 0.9169, -0.0937, 1.2906],
        [0.5432, 2.4151, -0.3814, 0.2830],
        [-0.0724, 0.7037, -0.1951, -0.1191]
      ])

    expected =
      ExTorch.tensor([
        [0.6804],
        [1.2957],
        [1.1984],
        [0.4193]
      ])

    out = ExTorch.std(input, -1, 1, true)
    assert ExTorch.allclose(out, expected, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "std/4 with kwargs" do
    input =
      ExTorch.tensor([
        [0.0686, 0.7169, 0.2143, 1.5755],
        [-1.6080, 0.9169, -0.0937, 1.2906],
        [0.5432, 2.4151, -0.3814, 0.2830],
        [-0.0724, 0.7037, -0.1951, -0.1191]
      ])

    expected =
      ExTorch.tensor([
        [0.6804],
        [1.2957],
        [1.1984],
        [0.4193]
      ])

    out = ExTorch.std(input, -1, 1, keepdim: true)
    assert ExTorch.allclose(out, expected, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "std/5" do
    input =
      ExTorch.tensor([
        [0.0686, 0.7169, 0.2143, 1.5755],
        [-1.6080, 0.9169, -0.0937, 1.2906],
        [0.5432, 2.4151, -0.3814, 0.2830],
        [-0.0724, 0.7037, -0.1951, -0.1191]
      ])

    expected =
      ExTorch.tensor([
        [0.6804],
        [1.2957],
        [1.1984],
        [0.4193]
      ])

    out = ExTorch.empty_like(expected)
    ExTorch.std(input, -1, 1, true, out)
    assert ExTorch.allclose(out, expected, rtol: 1.0e-3, atol: 1.0e-3)
  end

  test "std_mean/1" do
    input = ExTorch.randn({4, 4})
    expected_std = ExTorch.std(input)
    expected_mean = ExTorch.mean(input)
    {std, mean} = ExTorch.std_mean(input)

    assert ExTorch.allclose(std, expected_std)
    assert ExTorch.allclose(mean, expected_mean)
  end

  test "std_mean/2" do
    input = ExTorch.randn({4, 4})
    expected_std = ExTorch.std(input, -1)
    expected_mean = ExTorch.mean(input, -1)
    {std, mean} = ExTorch.std_mean(input, -1)

    assert ExTorch.allclose(std, expected_std)
    assert ExTorch.allclose(mean, expected_mean)
  end

  test "std_mean/2 with kwargs" do
    input = ExTorch.randn({4, 4})
    expected_std = ExTorch.std(input, keepdim: true)
    expected_mean = ExTorch.mean(input, keepdim: true)
    {std, mean} = ExTorch.std_mean(input, keepdim: true)

    assert ExTorch.allclose(std, expected_std)
    assert ExTorch.allclose(mean, expected_mean)
  end

  test "std_mean/3" do
    input = ExTorch.randn({4, 4})
    expected_std = ExTorch.std(input, -1, 2)
    expected_mean = ExTorch.mean(input, -1)
    {std, mean} = ExTorch.std_mean(input, -1, 2)

    assert ExTorch.allclose(std, expected_std)
    assert ExTorch.allclose(mean, expected_mean)
  end

  test "std_mean/3 with kwargs" do
    input = ExTorch.randn({4, 4})
    expected_std = ExTorch.std(input, -1, keepdim: true)
    expected_mean = ExTorch.mean(input, -1, keepdim: true)
    {std, mean} = ExTorch.std_mean(input, -1, keepdim: true)

    assert ExTorch.allclose(std, expected_std)
    assert ExTorch.allclose(mean, expected_mean)
  end

  test "std_mean/4" do
    input = ExTorch.randn({4, 4})
    expected_std = ExTorch.std(input, -1, 1, true)
    expected_mean = ExTorch.mean(input, -1, true)
    {std, mean} = ExTorch.std_mean(input, -1, 1, true)

    assert ExTorch.allclose(std, expected_std)
    assert ExTorch.allclose(mean, expected_mean)
  end

  test "std_mean/5" do
    input = ExTorch.randn({4, 4})
    expected_std = ExTorch.std(input, -1, 1, true)
    expected_mean = ExTorch.mean(input, -1, true)

    std = ExTorch.empty_like(expected_std)
    mean = ExTorch.empty_like(expected_mean)

    ExTorch.std_mean(input, -1, 1, true, {std, mean})

    assert ExTorch.allclose(std, expected_std)
    assert ExTorch.allclose(mean, expected_mean)
  end

  test "unique/1" do
    input =
      ExTorch.tensor(
        [
          [3, 3, -1, 2, 1],
          [1, 3, -1, 0, 0],
          [1, 1, 2, 2, 2],
          [2, 1, 3, -1, 0],
          [0, 1, 2, 3, 1]
        ],
        dtype: :int32
      )

    expected = ExTorch.tensor([-1, 0, 1, 2, 3], dtype: :int32)
    out = ExTorch.unique(input)
    assert ExTorch.equal(out, expected)
  end

  test "unique/2" do
    input =
      ExTorch.tensor(
        [
          [1, 0, 0, 0],
          [1, 1, 1, 0],
          [3, 3, 3, 1],
          [-1, 0, -1, -1]
        ],
        dtype: :int64
      )

    expected = ExTorch.tensor([3, -1, 0, 1], dtype: :int64)
    out = ExTorch.unique(input, false)
    assert ExTorch.equal(out, expected)
  end

  test "unique/2 with kwargs" do
    input =
      ExTorch.tensor(
        [
          [3, 3, -1, 2, 1],
          [1, 3, -1, 0, 0],
          [1, 1, 2, 2, 2],
          [2, 1, 3, -1, 0],
          [0, 1, 2, 3, 1]
        ],
        dtype: :int32
      )

    expected_unique = ExTorch.tensor([-1, 0, 1, 2, 3], dtype: :int32)
    expected_counts = ExTorch.tensor([3, 4, 7, 6, 5], dtype: :int64)
    {unique, counts} = ExTorch.unique(input, return_counts: true)
    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(counts, expected_counts)
  end

  test "unique/3" do
    input =
      ExTorch.tensor(
        [
          [1, 0, 0, 0],
          [1, 1, 1, 0],
          [3, 3, 3, 1],
          [-1, 0, -1, -1]
        ],
        dtype: :int64
      )

    expected_unique = ExTorch.tensor([-1, 0, 1, 3], dtype: :int64)

    expected_inverse =
      ExTorch.tensor(
        [
          [2, 1, 1, 1],
          [2, 2, 2, 1],
          [3, 3, 3, 2],
          [0, 1, 0, 0]
        ],
        dtype: :int64
      )

    {unique, inverse} = ExTorch.unique(input, true, true)
    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(inverse, expected_inverse)
  end

  test "unique/3 with kwargs" do
    input =
      ExTorch.tensor(
        [
          [1, 0, 0, 0],
          [1, 1, 1, 0],
          [3, 3, 3, 1],
          [-1, 0, -1, -1]
        ],
        dtype: :int64
      )

    expected_unique = ExTorch.tensor([-1, 0, 1, 3], dtype: :int64)

    expected_inverse =
      ExTorch.tensor(
        [
          [2, 1, 1, 1],
          [2, 2, 2, 1],
          [3, 3, 3, 2],
          [0, 1, 0, 0]
        ],
        dtype: :int64
      )

    {unique, inverse} = ExTorch.unique(input, true, return_inverse: true)
    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(inverse, expected_inverse)
  end

  test "unique/4" do
    input =
      ExTorch.tensor(
        [
          [1, 0, 0, 0],
          [1, 1, 1, 0],
          [3, 3, 3, 1],
          [-1, 0, -1, -1]
        ],
        dtype: :int64
      )

    expected_unique = ExTorch.tensor([-1, 0, 1, 3], dtype: :int64)

    expected_inverse =
      ExTorch.tensor(
        [
          [2, 1, 1, 1],
          [2, 2, 2, 1],
          [3, 3, 3, 2],
          [0, 1, 0, 0]
        ],
        dtype: :int64
      )

    expected_counts = ExTorch.tensor([3, 5, 5, 3], dtype: :int64)

    {unique, inverse, counts} = ExTorch.unique(input, true, true, true)
    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(inverse, expected_inverse)
    assert ExTorch.equal(counts, expected_counts)
  end

  test "unique/4 with kwargs" do
    input =
      ExTorch.tensor(
        [
          [1, 0, 0, 0],
          [1, 1, 1, 0],
          [3, 3, 3, 1],
          [-1, 0, -1, -1]
        ],
        dtype: :int64
      )

    expected_unique = ExTorch.tensor([-1, 0, 1, 3], dtype: :int64)
    expected_counts = ExTorch.tensor([3, 5, 5, 3], dtype: :int64)

    {unique, counts} = ExTorch.unique(input, true, false, return_counts: true)
    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(counts, expected_counts)
  end

  test "unique/5" do
    input =
      ExTorch.tensor(
        [
          [1, 0, 0, 0],
          [-1, 0, -1, -1],
          [1, 0, 0, 0],
          [-1, 0, -1, -1]
        ],
        dtype: :int64
      )

    expected_unique =
      ExTorch.tensor(
        [
          [-1, 0, -1, -1],
          [1, 0, 0, 0]
        ],
        dtype: :int64
      )

    expected_inverse = ExTorch.tensor([1, 0, 1, 0], dtype: :int64)
    expected_counts = ExTorch.tensor([2, 2], dtype: :int64)

    {unique, inverse, counts} = ExTorch.unique(input, true, true, true, 0)
    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(inverse, expected_inverse)
    assert ExTorch.equal(counts, expected_counts)
  end

  test "unique_consecutive/1" do
    input = ExTorch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype: :int32)
    expected_out = ExTorch.tensor([1, 2, 3, 1, 2], dtype: :int32)
    out = ExTorch.unique_consecutive(input)
    assert ExTorch.equal(out, expected_out)
  end

  test "unique_consecutive/2" do
    input = ExTorch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype: :int32)
    expected_unique = ExTorch.tensor([1, 2, 3, 1, 2], dtype: :int32)
    expected_inverse = ExTorch.tensor([0, 0, 1, 1, 2, 3, 3, 4], dtype: :int64)
    {unique, inverse} = ExTorch.unique_consecutive(input, true)

    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(inverse, expected_inverse)
  end

  test "unique_consecutive/2 with kwargs" do
    input = ExTorch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype: :int32)
    expected_unique = ExTorch.tensor([1, 2, 3, 1, 2], dtype: :int32)
    expected_counts = ExTorch.tensor([2, 2, 1, 2, 1], dtype: :int64)
    {unique, counts} = ExTorch.unique_consecutive(input, return_counts: true)

    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(counts, expected_counts)
  end

  test "unique_consecutive/3" do
    input = ExTorch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype: :int32)
    expected_unique = ExTorch.tensor([1, 2, 3, 1, 2], dtype: :int32)
    expected_inverse = ExTorch.tensor([0, 0, 1, 1, 2, 3, 3, 4], dtype: :int64)
    expected_counts = ExTorch.tensor([2, 2, 1, 2, 1], dtype: :int64)
    {unique, inverse, counts} = ExTorch.unique_consecutive(input, true, true)

    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(inverse, expected_inverse)
    assert ExTorch.equal(counts, expected_counts)
  end

  test "unique_consecutive/3 with kwargs" do
    input = ExTorch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype: :int32)
    expected_unique = ExTorch.tensor([1, 2, 3, 1, 2], dtype: :int32)
    expected_inverse = ExTorch.tensor([0, 0, 1, 1, 2, 3, 3, 4], dtype: :int64)
    expected_counts = ExTorch.tensor([2, 2, 1, 2, 1], dtype: :int64)
    {unique, inverse, counts} = ExTorch.unique_consecutive(input, true, return_counts: true)

    assert ExTorch.equal(unique, expected_unique)
    assert ExTorch.equal(inverse, expected_inverse)
    assert ExTorch.equal(counts, expected_counts)
  end

  test "unique_consecutive/4" do
    input =
      ExTorch.tensor(
        [
          [-1, -1, -1],
          [0, 1, 1],
          [0, 1, 1],
          [-1, -1, -1]
        ],
        dtype: :int64
      )

    expected_unique =
      ExTorch.tensor(
        [
          [-1, -1, -1],
          [0, 1, 1],
          [-1, -1, -1]
        ],
        dtype: :int32
      )

    unique = ExTorch.unique_consecutive(input, false, false, 0)
    assert ExTorch.equal(unique, expected_unique)
  end
end
