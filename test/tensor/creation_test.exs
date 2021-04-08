defmodule ExTorchTest.Tensor.CreationTest do
  use ExUnit.Case
  # doctest ExTorch

  test "zeros/1" do
    tensor = ExTorch.zeros({2, 3, 4, 5})
    assert tensor.size == {2, 3, 4, 5}
    assert tensor.dtype == :float32
    assert tensor.device == :cpu

    assert ExTorch.size(tensor) == tensor.size
  end

  test "zeros with dtype" do
    tensor = ExTorch.zeros({2, 3, 4, 5}, dtype: :float64)
    assert tensor.size == {2, 3, 4, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.size(tensor) == tensor.size
  end

  test "empty/1" do
    tensor = ExTorch.empty({5, 6, 7})
    assert tensor.size == {5, 6, 7}
    assert tensor.dtype == :float32
    assert tensor.device == :cpu

    assert ExTorch.size(tensor) == tensor.size
  end

  test "empty with dtype" do
    tensor = ExTorch.empty({5, 6, 7}, dtype: :int64)
    assert tensor.size == {5, 6, 7}
    assert tensor.dtype == :long
    assert tensor.device == :cpu

    assert ExTorch.size(tensor) == tensor.size
  end

  test "ones/1" do
    tensor = ExTorch.ones({8, 10, 2, 1})
    assert tensor.size == {8, 10, 2, 1}
    assert tensor.dtype == :float32
    assert tensor.device == :cpu

    assert ExTorch.size(tensor) == tensor.size
  end

  test "ones with dtype" do
    tensor = ExTorch.ones({8, 10, 2, 1}, dtype: :uint8)
    assert tensor.size == {8, 10, 2, 1}
    assert tensor.dtype == :byte
    assert tensor.device == :cpu

    assert ExTorch.size(tensor) == tensor.size
  end

  test "rand/1" do
    tensor = ExTorch.rand({1, 3, 1, 5})
    assert tensor.size == {1, 3, 1, 5}
    assert tensor.dtype == :float32
    assert tensor.device == :cpu

    assert ExTorch.size(tensor) == tensor.size
  end

  test "rand with dtype" do
    tensor = ExTorch.rand({1, 3, 1, 5}, :float64)
    assert tensor.size == {1, 3, 1, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.size(tensor) == tensor.size
  end

end
