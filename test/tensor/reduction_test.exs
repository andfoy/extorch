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

end
