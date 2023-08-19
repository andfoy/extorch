defmodule ExTorchTest.Tensor.CreationTest do
  use ExUnit.Case
  # doctest ExTorch

  test "zeros/1" do
    tensor = ExTorch.zeros({2, 3, 4, 5})
    assert tensor.size == {2, 3, 4, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "zeros with dtype" do
    tensor = ExTorch.zeros({2, 3, 4, 5}, dtype: :float64)
    assert tensor.size == {2, 3, 4, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "empty/1" do
    tensor = ExTorch.empty({5, 6, 7})
    assert tensor.size == {5, 6, 7}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "empty with dtype" do
    tensor = ExTorch.empty({5, 6, 7}, dtype: :int64)
    assert tensor.size == {5, 6, 7}
    assert tensor.dtype == :long
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "ones/1" do
    tensor = ExTorch.ones({8, 10, 2, 1})
    assert tensor.size == {8, 10, 2, 1}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "ones with dtype" do
    tensor = ExTorch.ones({8, 10, 2, 1}, dtype: :uint8)
    assert tensor.size == {8, 10, 2, 1}
    assert tensor.dtype == :byte
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "rand/1" do
    tensor = ExTorch.rand({1, 3, 1, 5})
    assert tensor.size == {1, 3, 1, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "rand with dtype" do
    tensor = ExTorch.rand({1, 3, 1, 5}, dtype: :float64)
    assert tensor.size == {1, 3, 1, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "randn/1" do
    tensor = ExTorch.randn({2, 2, 3, 5, 1})
    assert tensor.size == {2, 2, 3, 5, 1}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "randn with dtype" do
    tensor = ExTorch.randn({2, 2, 3, 5, 1}, dtype: :float64)
    assert tensor.size == {2, 2, 3, 5, 1}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "randint/2" do
    tensor = ExTorch.randint(5, {3, 3, 3})
    assert tensor.size == {3, 3, 3}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "randint/3" do
    tensor = ExTorch.randint(-3, 10, {3, 3, 3})
    assert tensor.size == {3, 3, 3}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "randint/3 with dtype" do
    tensor = ExTorch.randint(10, {3, 3, 3}, dtype: :float64)
    assert tensor.size == {3, 3, 3}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "randint/4 with kwargs" do
    tensor = ExTorch.randint(10, {3, 3, 3}, dtype: :float64, requires_grad: true)
    assert tensor.size == {3, 3, 3}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "randint/4" do
    tensor = ExTorch.randint(-3, 10, {3, 3, 3}, dtype: :float64)
    assert tensor.size == {3, 3, 3}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/1" do
    tensor = ExTorch.arange(10)
    assert tensor.size == {10}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/1 with kwargs" do
    tensor = ExTorch.arange(10, dtype: :float64)
    assert tensor.size == {10}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/2" do
    tensor = ExTorch.arange(-2, 10)
    assert tensor.size == {12}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/2 with kwargs" do
    tensor = ExTorch.arange(-2, 10, dtype: :uint8)
    assert tensor.size == {12}
    assert tensor.dtype == :byte
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/3 with dtype" do
    tensor = ExTorch.arange(1, 10, dtype: :uint8)
    assert tensor.size == {9}
    assert tensor.dtype == :byte
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/3" do
    tensor = ExTorch.arange(0, 0.5, 0.1)
    assert tensor.size == {5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/4 with dtype and kwargs" do
    tensor = ExTorch.arange(0.5, step: 0.1, dtype: :float64, requires_grad: true)
    assert tensor.size == {5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/4 with dtype" do
    tensor = ExTorch.arange(-0.5, 0.5, 0.1, dtype: :float64)
    assert tensor.size == {10}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "arange/4 with kwargs" do
    tensor = ExTorch.arange(-0.5, 0.5, 0.1, dtype: :float64)
    assert tensor.size == {10}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "eye/1" do
    tensor = ExTorch.eye(3)
    assert tensor.size == {3, 3}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "eye/1 with kwargs" do
    tensor = ExTorch.eye(3, memory_format: :contiguous, dtype: :uint8)
    assert tensor.size == {3, 3}
    assert tensor.dtype == :byte
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "eye/2" do
    tensor = ExTorch.eye(3, 5)
    assert tensor.size == {3, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "eye/3 with kwargs" do
    tensor = ExTorch.eye(3, 5, dtype: :int32, requires_grad: false)
    assert tensor.size == {3, 5}
    assert tensor.dtype == :int
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "eye/3" do
    tensor = ExTorch.eye(3, 5, dtype: :int64)
    assert tensor.size == {3, 5}
    assert tensor.dtype == :long
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "full/2" do
    tensor = ExTorch.full({3, 4, 5}, 5)
    assert tensor.size == {3, 4, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "full/3" do
    tensor = ExTorch.full({3, 4, 5}, -1.456, dtype: :float64)
    assert tensor.size == {3, 4, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "full/3 with kwargs" do
    tensor = ExTorch.full({3, 4, 5}, 0.003, requires_grad: true)
    assert tensor.size == {3, 4, 5}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "linspace/3" do
    tensor = ExTorch.linspace(-1.5, 2.6, 10)
    assert tensor.size == {10}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "linspace/4" do
    tensor = ExTorch.linspace(-1.5, 2.6, 20, dtype: :float64)
    assert tensor.size == {20}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "linspace/4 with kwargs" do
    tensor = ExTorch.linspace(-0.5, 5, 20, requires_grad: true)
    assert tensor.size == {20}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "logspace/3" do
    tensor = ExTorch.logspace(-1.5, 2.6, 10)
    assert tensor.size == {10}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "logspace/4" do
    tensor = ExTorch.logspace(-1.5, 2.6, 20, 2)
    assert tensor.size == {20}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "logspace/4 with kwargs" do
    tensor = ExTorch.logspace(-0.5, 5, 20, requires_grad: true)
    assert tensor.size == {20}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "tensor/1 with byte" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert tensor.size == {3, 3}
    assert tensor.dtype == :byte
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "tensor/1 with int32" do
    tensor =
      ExTorch.tensor([
        [[6, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[-6, -1, -2], [-3, -4, -5], [-6, -7, -8]]
      ])

    assert tensor.size == {2, 3, 3}
    assert tensor.dtype == :int
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "tensor/1 with float32" do
    tensor =
      ExTorch.tensor(
        [[[6, 1, 2], [3, 4, 5], [6, 7, 8]], [[-6, -1, -2], [-3, -4, -5], [-6, -7, -8]]],
        dtype: :float32
      )

    assert tensor.size == {2, 3, 3}
    assert tensor.dtype == :float
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "tensor/1 with float64" do
    tensor =
      ExTorch.tensor([
        [[6, 1.0, 2], [3, 4, 5], [6, 7, 8]],
        [[-6, -1, -2], [-3, -4, -5], [-6, -7, -8]]
      ])

    assert tensor.size == {2, 3, 3}
    assert tensor.dtype == :double
    assert tensor.device == :cpu

    assert ExTorch.Tensor.size(tensor) == tensor.size
  end

  test "tensor/1 with ExTorch.Complex" do
    tensor = ExTorch.tensor([ExTorch.Complex.complex(-1, 2), 7, ExTorch.Complex.complex(8, 4)])
    assert tensor.size == {3}
    assert tensor.dtype == :complex_float
    assert tensor.device == :cpu

    real_part = [-1.0, 7.0, 8.0]

    tensor_real =
      tensor
      |> ExTorch.real()
      |> ExTorch.Tensor.to_list()

    imag_part = [2.0, 0.0, 4.0]

    tensor_imag =
      tensor
      |> ExTorch.imag()
      |> ExTorch.Tensor.to_list()

    assert tensor_real == real_part
    assert tensor_imag == imag_part
  end
end
