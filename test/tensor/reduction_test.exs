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
end
