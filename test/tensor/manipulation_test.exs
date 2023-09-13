defmodule ExTorchTest.Tensor.ManipulationTest do
  use ExUnit.Case

  test "unsqueeze(0)" do
    tensor = ExTorch.empty({3, 3})
    unsqueezed_tensor = ExTorch.unsqueeze(tensor, 0)
    assert unsqueezed_tensor.size == {1, 3, 3}
  end

  test "unsqueeze with random positive dimension" do
    dims = {3, 6, 7, 8, 5}
    rand_dim = Enum.random(0..4)

    expected_dims =
      dims
      |> Tuple.to_list()
      |> List.insert_at(rand_dim, 1)
      |> List.to_tuple()

    tensor = ExTorch.empty(dims)
    unsqueezed_tensor = ExTorch.unsqueeze(tensor, rand_dim)
    assert unsqueezed_tensor.size == expected_dims
  end

  test "unsqueeze with random negative dimension" do
    dims = {3, 6, 7, 8, 5}
    rand_dim = Enum.random(-1..-5)
    pos_dim = 6 + rand_dim

    expected_dims =
      dims
      |> Tuple.to_list()
      |> List.insert_at(pos_dim, 1)
      |> List.to_tuple()

    tensor = ExTorch.empty(dims)
    unsqueezed_tensor = ExTorch.unsqueeze(tensor, rand_dim)
    assert unsqueezed_tensor.size == expected_dims
  end

  test "reshape/2 with negative value" do
    tensor = ExTorch.empty({3, 4, 5, 2})
    reshaped_tensor = ExTorch.reshape(tensor, {3, -1, 2})
    assert reshaped_tensor.size == {3, 20, 2}
  end

  test "reshape/2" do
    tensor = ExTorch.empty({3, 4, 5, 2})
    reshaped_tensor = ExTorch.reshape(tensor, {20, 6})
    assert reshaped_tensor.size == {20, 6}
  end

  test "conj/1" do
    input = [ExTorch.Complex.complex(-1, 5), ExTorch.Complex.complex(2.3, -4)]
    a = ExTorch.tensor(input)
    b = ExTorch.conj(a)

    assert ExTorch.Tensor.is_conj(b)
    back = ExTorch.Tensor.to_list(b)
    signs = [1, -1]

    back_signs =
      Enum.map(
        back,
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
