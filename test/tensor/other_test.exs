defmodule ExTorchTest.Tensor.OtherTest do
  use ExUnit.Case

  test "test view_as_complex/1" do
    input = ExTorch.rand({3, 4, 2})
    complex_view = ExTorch.view_as_complex(input)

    assert complex_view.size == {3, 4}
    assert complex_view.dtype == :complex_double

    real_part =
      input
      |> ExTorch.index([:..., 0])
      |> ExTorch.Tensor.to_list()

    imag_part =
      input
      |> ExTorch.index([:..., 1])
      |> ExTorch.Tensor.to_list()

    complex_view_real =
      complex_view
      |> ExTorch.real()
      |> ExTorch.Tensor.to_list()

    complex_view_imag =
      complex_view
      |> ExTorch.imag()
      |> ExTorch.Tensor.to_list()

    assert complex_view_real == real_part
    assert complex_view_imag == imag_part
  end
end
