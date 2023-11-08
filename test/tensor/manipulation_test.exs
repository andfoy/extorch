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

  test "transpose/3" do
    input = ExTorch.tensor([[0, 1, 2], [3, 4, 5]], dtype: :int32)
    expected = ExTorch.tensor([[0, 3], [1, 4], [2, 5]], dtype: :int32)
    out = ExTorch.transpose(input, 0, 1)
    assert ExTorch.equal(out, expected)
  end

  test "adjoint/1" do
    input = ExTorch.rand({3, 3}, dtype: :complex64)
    expected = ExTorch.conj(ExTorch.transpose(input, 0, 1))
    out = ExTorch.adjoint(input)
    assert ExTorch.allclose(out, expected)
  end

  test "cat/1" do
    input = ExTorch.rand({5, 2})
    out = ExTorch.cat([input, input])
    assert out.size == {10, 2}
  end

  test "cat/2" do
    input = ExTorch.rand({5, 2})
    out = ExTorch.cat([input, input], -1)
    assert out.size == {5, 4}
  end

  test "cat/2 with kwargs" do
    input = ExTorch.rand({5, 2})
    out = ExTorch.cat([input, input], dim: -1)
    assert out.size == {5, 4}
  end

  test "cat/3" do
    input = ExTorch.ones({3, 4})
    expected = ExTorch.ones({3, 8})
    out = ExTorch.empty({3, 8})

    ExTorch.cat([input, input], -1, out)
    assert ExTorch.allclose(out, expected)
  end

  test "chunk/2" do
    input = ExTorch.arange(11)
    out = ExTorch.chunk(input, 6)
    assert length(out) == 6
  end

  test "chunk/3" do
    input = ExTorch.rand({3, 6})
    out = ExTorch.chunk(input, 3, -1)
    expected = [{3, 2}, {3, 2}, {3, 2}]
    sizes = for t <- out, do: t.size
    assert sizes == expected
  end

  test "tensor_split/2" do
    input = ExTorch.arange(10)
    out = ExTorch.tensor_split(input, 2)
    expected = [{5}, {5}]
    sizes = for t <- out, do: t.size
    assert sizes == expected
  end

  test "tensor_split/2 with sections" do
    input = ExTorch.arange(10)
    out = ExTorch.tensor_split(input, [2, 5])
    expected = [{2}, {3}, {5}]
    sizes = for t <- out, do: t.size
    assert sizes == expected
  end

  test "tensor_split/3" do
    input = ExTorch.rand({4, 6})
    out = ExTorch.tensor_split(input, 2, 1)
    expected = [{4, 3}, {4, 3}]
    sizes = for t <- out, do: t.size
    assert sizes == expected
  end

  test "dsplit/2" do
    input = ExTorch.arange(16) |> ExTorch.reshape({2, 2, 4})
    out = ExTorch.dsplit(input, 2)
    expected = [{2, 2, 2}, {2, 2, 2}]
    sizes = for t <- out, do: t.size
    assert sizes == expected
  end

  test "dsplit/2 with list" do
    input = ExTorch.rand({2, 2, 10})
    out = ExTorch.dsplit(input, [2, 7])
    expected = [{2, 2, 2}, {2, 2, 5}, {2, 2, 3}]
    sizes = for t <- out, do: t.size
    assert sizes == expected
  end

  test "column_stack/1" do
    a = ExTorch.rand({3})
    b = ExTorch.rand({3})
    out = ExTorch.column_stack({a, b})
    expected = {3, 2}
    assert out.size == expected
  end

  test "column_stack/2" do
    a = ExTorch.rand({3, 2})
    b = ExTorch.rand({3, 7})
    c = ExTorch.rand({3})
    out = ExTorch.empty({3, 10})

    ExTorch.column_stack([a, b, c], out)
  end

  test "dstack/1" do
    a = ExTorch.rand({4})
    b = ExTorch.rand({4})
    values_3d = for v <- [a, b], do: ExTorch.unsqueeze(v, 0) |> ExTorch.unsqueeze(-1)

    expected = ExTorch.concat(values_3d, -1)
    out = ExTorch.dstack([a, b])
    assert ExTorch.allclose(out, expected)
  end

  test "dstack/2" do
    a = ExTorch.rand({4, 2})
    b = ExTorch.rand({4, 2})
    values_3d = for v <- [a, b], do: ExTorch.unsqueeze(v, -1)

    expected = ExTorch.concat(values_3d, -1)
    out = ExTorch.empty_like(expected)
    ExTorch.dstack([a, b], out)
    assert ExTorch.allclose(out, expected)
  end

  test "gather/3" do
    input =
      ExTorch.tensor([
        [1, 2],
        [3, 4]
      ])

    expected =
      ExTorch.tensor([
        [2, 1],
        [4, 3]
      ])

    index =
      ExTorch.tensor(
        [
          [1, 0],
          [1, 0]
        ],
        dtype: :int64
      )

    out = ExTorch.gather(input, -1, index)
    assert ExTorch.equal(out, expected)
  end

  test "gather/5" do
    input =
      ExTorch.tensor([
        [1, 2],
        [3, 4]
      ])

    expected =
      ExTorch.tensor([
        [3, 4],
        [1, 2]
      ])

    index =
      ExTorch.tensor(
        [
          [1, 1],
          [0, 0]
        ],
        dtype: :int64
      )

    out = ExTorch.empty_like(input)
    ExTorch.gather(input, 0, index, false, out)
    assert ExTorch.equal(out, expected)
  end

  test "hsplit/2" do
    input = ExTorch.rand({4, 6})
    out = ExTorch.hsplit(input, 2)
    expected = [{4, 3}, {4, 3}]
    sizes = for t <- out, do: t.size
    assert sizes == expected
  end

  test "hsplit/2 with list" do
    input = ExTorch.rand({2, 10, 2})
    out = ExTorch.hsplit(input, [2, 7])
    expected = [{2, 2, 2}, {2, 5, 2}, {2, 3, 2}]
    sizes = for t <- out, do: t.size
    assert sizes == expected
  end

  test "hstack/1" do
    input = ExTorch.rand({2, 10, 2})
    split = ExTorch.hsplit(input, [2, 7])
    out = ExTorch.hstack(split)
    assert ExTorch.allclose(out, input)
  end

  test "hstack/2" do
    input = ExTorch.rand({2, 10, 2})
    split = ExTorch.hsplit(input, [2, 7])
    out = ExTorch.empty_like(input)

    ExTorch.hstack(split, out)
    assert ExTorch.allclose(out, input)
  end

  test "movedim/3" do
    input = ExTorch.empty({3, 4, 5})
    out = ExTorch.movedim(input, 0, 1)
    assert out.size == {4, 3, 5}
  end

  test "movedim/3 with tuple" do
    input = ExTorch.empty({3, 4, 5, 1})
    out = ExTorch.movedim(input, {0, 2}, {1, -1})
    assert out.size == {4, 3, 1, 5}
  end
end
