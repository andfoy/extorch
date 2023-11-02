defmodule ExTorchTest.Tensor.InfoTest do
  use ExUnit.Case

  test "size/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.Tensor.size(tensor) == {3, 3}
  end

  test "dim/1" do
    tensor = ExTorch.randn({2, 1, 3, 2})
    assert ExTorch.Tensor.dim(tensor) == 4
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

    assert ExTorch.Tensor.repr(tensor) ==
             "[[5.0000e-10, 5.0000e-10, 5.0000e-10, ..., 5.0000e-10, 5.0000e-10,\n  5.0000e-10],\n" <>
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

  test "numel/1" do
    tensor = ExTorch.empty({3, 4, 5})
    assert ExTorch.Tensor.numel(tensor) == 3 * 4 * 5
  end

  test "memory_format/0" do
    tensor = ExTorch.empty({3, 4, 5})
    assert ExTorch.Tensor.memory_format(tensor) == :contiguous

    tensor = ExTorch.empty({1, 3, 4, 5}, memory_format: :channels_last)
    assert ExTorch.Tensor.memory_format(tensor) == :channels_last
  end

  test "layout/0" do
    tensor = ExTorch.empty({3, 4, 5})
    assert ExTorch.Tensor.layout(tensor) == :strided
  end

  test "is_complex/1" do
    tensor = ExTorch.empty({2, 2}, dtype: :complex64)
    assert ExTorch.Tensor.is_complex(tensor)

    tensor = ExTorch.empty({2, 2})
    assert !ExTorch.Tensor.is_complex(tensor)
  end

  test "is_floating_point/1" do
    tensor = ExTorch.empty({2, 3})
    assert ExTorch.Tensor.is_floating_point(tensor)

    tensor = ExTorch.empty({3, 4}, dtype: :float64)
    assert ExTorch.Tensor.is_floating_point(tensor)

    tensor = ExTorch.empty({3, 4}, dtype: :int32)
    assert !ExTorch.Tensor.is_floating_point(tensor)
  end

  test "is_conj/1" do
    tensor = ExTorch.empty({2, 2})
    assert !ExTorch.Tensor.is_conj(tensor)

    tensor = ExTorch.empty({2, 2}, dtype: :complex128)
    assert !ExTorch.Tensor.is_conj(tensor)

    conj = ExTorch.conj(tensor)
    assert ExTorch.Tensor.is_conj(conj)

    materialized = ExTorch.resolve_conj(conj)
    assert !ExTorch.Tensor.is_conj(materialized)
  end

  test "is_nonzero/1" do
    assert !ExTorch.Tensor.is_nonzero(ExTorch.tensor([0.0]))
    assert ExTorch.Tensor.is_nonzero(ExTorch.tensor([1.5]))
    assert !ExTorch.Tensor.is_nonzero(ExTorch.tensor([false]))
    assert ExTorch.Tensor.is_nonzero(ExTorch.tensor([3]))
  end

  test "item/1" do
    a = ExTorch.tensor(false)
    assert !ExTorch.Tensor.item(a)

    a = ExTorch.tensor(3)
    assert ExTorch.Tensor.item(a) == 3

    a = ExTorch.tensor(-3.5)
    assert ExTorch.Tensor.item(a) == -3.5

    a = ExTorch.tensor(:inf)
    assert ExTorch.Tensor.item(a) == :inf

    a = ExTorch.tensor(:ninf)
    assert ExTorch.Tensor.item(a) == :ninf

    a = ExTorch.tensor(:nan)
    assert ExTorch.Tensor.item(a) == :nan

    a = ExTorch.tensor(ExTorch.Complex.complex(-3, 2))
    assert ExTorch.Tensor.item(a) == ExTorch.Complex.complex(-3, 2)
  end

  test "to/1" do
    a = ExTorch.rand({3, 3})
    b = ExTorch.Tensor.to(a)
    a = ExTorch.index_put(a, [0, 0], 1, inplace: true)
    assert ExTorch.equal(a, b)
  end

  test "to/2 with copy" do
    a = ExTorch.rand({3, 3})
    b = ExTorch.Tensor.to(a, copy: true)
    a = ExTorch.index_put(a, [0, 0], 1)
    assert !ExTorch.equal(a, b)
  end

  test "to/2 with dtype" do
    a = ExTorch.rand({3, 3})
    b = ExTorch.Tensor.to(a, dtype: :complex64)
    assert ExTorch.equal(a, ExTorch.real(b))
  end

  test "to/2 with device" do
    nvcc = System.find_executable("nvcc")

    case nvcc do
      nil ->
        nil

      _ ->
        a = ExTorch.rand({3, 3})
        b = ExTorch.Tensor.to(a, device: :cuda)
        assert b.device == {:cuda, 0}
    end
  end
end
