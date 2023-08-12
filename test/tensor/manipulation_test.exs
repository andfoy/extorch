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

  test "index tensor with an integer" do
    x = ExTorch.rand({3, 4, 5})
    indexed = ExTorch.index(x, 0)
    assert indexed.size == {4, 5}
  end

  test "index tensor with an integer sequence" do
    x = ExTorch.rand({3, 4, 5})
    indexed = ExTorch.index(x, [0, 1])
    assert indexed.size == {5}
  end

  test "index tensor with a slice" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, ExTorch.slice(1))
    assert indexed.size == {5, 4, 5}

    indexed = ExTorch.index(x, ExTorch.slice(nil, -1))
    assert indexed.size == {5, 4, 5}

    indexed = ExTorch.index(x, ExTorch.slice(nil, nil, 2))
    assert indexed.size == {3, 4, 5}

    indexed = ExTorch.index(x, ExTorch.slice())
    assert indexed.size == {6, 4, 5}

    indexed = ExTorch.index(x, :::)
    assert indexed.size == {6, 4, 5}

    indexed = ExTorch.index(x, 1..6)
    assert indexed.size == {5, 4, 5}

    indexed = ExTorch.index(x, 0..6//2)
    assert indexed.size == {3, 4, 5}
  end

  test "index tensor with a sequence of slices" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, [ExTorch.slice(), ExTorch.slice(0, -2)])
    assert indexed.size == {6, 2, 5}

    indexed = ExTorch.index(x, [ExTorch.slice(1), :::, 0..4])
    assert indexed.size == {5, 4, 4}
  end

  test "index tensor with ellipsis" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, :...)
    assert indexed.size == {6, 4, 5}

    indexed = ExTorch.index(x, :ellipsis)
    assert indexed.size == {6, 4, 5}
  end

  test "index tensor with ellipsis and other index" do
    x = ExTorch.rand({6, 4, 2, 5})
    indexed = ExTorch.index(x, [:..., 0])
    assert indexed.size == {6, 4, 2}

    indexed = ExTorch.index(x, {:..., 0})
    assert indexed.size == {6, 4, 2}

    indexed = ExTorch.index(x, [1, :ellipsis, ExTorch.slice(0, 3)])
    assert indexed.size == {4, 2, 3}
  end

  test "index tensor with nil" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, nil)
    assert indexed.size == {1, 6, 4, 5}
  end

  test "index tensor with boolean" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, true)
    assert indexed.size == {1, 6, 4, 5}

    indexed = ExTorch.index(x, false)
    assert indexed.size == {0, 6, 4, 5}
  end

  test "index tensor with a list of integers" do
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, [[0, 4]])
    assert indexed.size == {2, 4, 5}

    indexed = ExTorch.index(x, [[-1, -2, 3]])
    assert indexed.size == {3, 4, 5}

    indexed = ExTorch.index(x, [{5}])
    assert indexed.size == {1, 4, 5}
  end

  test "index tensor with another tensor" do
    i = ExTorch.tensor([[5, 1], [4, 1], [3, 1], [2, 1], [1, 1], [0, 1]], dtype: :int64)
    x = ExTorch.rand({6, 4, 5})
    indexed = ExTorch.index(x, i)
    assert indexed.size == {6, 2, 4, 5}
  end

end
