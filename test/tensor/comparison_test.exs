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


end
