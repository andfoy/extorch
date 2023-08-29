defmodule ExTorchTest.Tensor.InfoTest do
  use ExUnit.Case

  test "size/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.Tensor.size(tensor) == {3, 3}
  end

  test "dtype/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.Tensor.dtype(tensor) == :byte

    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype: :int64)
    assert ExTorch.Tensor.dtype(tensor) == :long
  end

  test "device/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.Tensor.device(tensor) == :cpu

    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], device: {:cpu, 0})
    assert ExTorch.Tensor.device(tensor) == :cpu
  end

  test "repr/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.Tensor.repr(tensor) == "[[0, 1, 2],\n [3, 4, 5],\n [6, 7, 8]]"

    tensor = ExTorch.tensor([[0.1, 1, 2], [3, -4, 5], [6, 7, 8]])

    assert ExTorch.Tensor.repr(tensor) ==
             "[[ 0.1000,  1.0000,  2.0000],\n [ 3.0000, -4.0000,  5.0000],\n [ 6.0000,  7.0000,  8.0000]]"

    tensor = ExTorch.full({2, 2, 3}, 3.1459)

    assert ExTorch.Tensor.repr(tensor) ==
             "[[[3.1459, 3.1459, 3.1459],\n  [3.1459, 3.1459, 3.1459]],\n\n [[3.1459, 3.1459, 3.1459],\n  [3.1459, 3.1459, 3.1459]]]"

    tensor = ExTorch.full({300, 10}, 0.0000000005)
    assert ExTorch.Tensor.repr(tensor) == "[[5.0000e-10, 5.0000e-10, 5.0000e-10, ..., 5.0000e-10, 5.0000e-10,\n  5.0000e-10],\n" <>
                                          " [5.0000e-10, 5.0000e-10, 5.0000e-10, ..., 5.0000e-10, 5.0000e-10,\n  5.0000e-10],\n" <>
                                          " [5.0000e-10, 5.0000e-10, 5.0000e-10, ..., 5.0000e-10, 5.0000e-10,\n  5.0000e-10],\n" <>
                                          " ...,\n" <>
                                          " [5.0000e-10, 5.0000e-10, 5.0000e-10, ..., 5.0000e-10, 5.0000e-10,\n  5.0000e-10],\n" <>
                                          " [5.0000e-10, 5.0000e-10, 5.0000e-10, ..., 5.0000e-10, 5.0000e-10,\n  5.0000e-10],\n" <>
                                          " [5.0000e-10, 5.0000e-10, 5.0000e-10, ..., 5.0000e-10, 5.0000e-10,\n  5.0000e-10]]"

    tensor = ExTorch.tensor([true, false])
    assert ExTorch.Tensor.repr(tensor) == "[ true, false]"
  end

  test "to_list/1" do
    tensor_info = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    tensor = ExTorch.tensor(tensor_info)
    assert ExTorch.Tensor.to_list(tensor) == tensor_info
  end

  test "to_list/1 come back" do
    # tensor_info = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # tensor = ExTorch.tensor(tensor_info)
    tensor = ExTorch.ones({3, 4, 5})
    l = ExTorch.Tensor.to_list(tensor)
    lt = ExTorch.tensor(l)
    assert ExTorch.Tensor.to_list(lt) == l
  end

  test "requires_grad/1" do
    tensor = ExTorch.empty({2}, requires_grad: true)
    assert ExTorch.Tensor.requires_grad(tensor)

    tensor = ExTorch.empty({2})
    assert !ExTorch.Tensor.requires_grad(tensor)
  end
end
