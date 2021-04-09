defmodule ExTorchTest.Tensor.InfoTest do
  use ExUnit.Case

  test "size/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.size(tensor) == {3, 3}
  end

  test "dtype/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.dtype(tensor) == :byte

    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype: :int64)
    assert ExTorch.dtype(tensor) == :long
  end

  test "device/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.device(tensor) == :cpu

    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], device: {:cpu, 0})
    assert ExTorch.device(tensor) == :cpu
  end

  test "repr/1" do
    tensor = ExTorch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert ExTorch.repr(tensor) == " 0  1  2\n 3  4  5\n 6  7  8\n[ CPUByteType{3,3} ]"

    tensor = ExTorch.tensor([[0.1, 1, 2], [3, -4, 5], [6, 7, 8]])

    assert ExTorch.repr(tensor) ==
             " 0.1000  1.0000  2.0000\n 3.0000 -4.0000  5.0000\n 6.0000  7.0000  8.0000\n[ CPUDoubleType{3,3} ]"

    tensor = ExTorch.full({2, 2, 3}, 3.1459)

    assert ExTorch.repr(tensor) ==
             "(1,.,.) = \n  3.1459  3.1459  3.1459\n  3.1459  3.1459  3.1459\n\n(2,.,.) = \n  3.1459  3.1459  3.1459\n  3.1459  3.1459  3.1459\n[ CPUFloatType{2,2,3} ]"
  end

  test "to_list/1" do
    tensor_info = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    tensor = ExTorch.tensor(tensor_info)
    assert ExTorch.to_list(tensor) == tensor_info
  end
end
