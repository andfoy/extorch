defmodule ExTorchTest.Tensor.MutationTest do
  use ExUnit.Case

  test "conj/1" do
    input = [ExTorch.Complex.complex(-1, 5), ExTorch.Complex.complex(2.3, -4)]
    a = ExTorch.tensor(input)
    b = ExTorch.conj(a)

    assert ExTorch.Tensor.is_conj(b)
    back = ExTorch.Tensor.to_list(b)
    signs = [1, -1]
    back_signs = Enum.map(back,
      fn %ExTorch.Complex{imaginary: imag} ->
        case imag >= 0 do
          true -> 1
          false -> -1
        end
      end
    )

    assert signs == back_signs
  end

end
