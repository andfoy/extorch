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
end
