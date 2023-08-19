defmodule ExTorchTest.Tensor.PointWiseTest do
  use ExUnit.Case

  test "test real/1" do
    real_part = ExTorch.rand({3, 3})
    imag_part = ExTorch.rand({3, 3})
    complex_tensor = ExTorch.complex(real_part, imag_part)

    real_list = ExTorch.Tensor.to_list(real_part)
    complex_real =
      complex_tensor
      |> ExTorch.real()
      |> ExTorch.Tensor.to_list()

    assert complex_real == real_list
  end

  test "test imag/1" do
    real_part = ExTorch.rand({3, 3})
    imag_part = ExTorch.rand({3, 3})
    complex_tensor = ExTorch.complex(real_part, imag_part)

    imag_list = ExTorch.Tensor.to_list(imag_part)
    complex_imag =
      complex_tensor
      |> ExTorch.imag()
      |> ExTorch.Tensor.to_list()

    assert complex_imag == imag_list
  end
end
