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

  test "narrow_copy/4" do
    input = ExTorch.randn({4, 3})
    expected = input[0..2]
    out = ExTorch.narrow_copy(input, 0, 0, 2)
    assert ExTorch.allclose(out, expected)
  end

  test "narrow_copy/5" do
    input = ExTorch.randn({4, 3})
    expected = input[{:"::", 1..3}]
    out = ExTorch.empty_like(expected)

    ExTorch.narrow_copy(input, 1, 1, 2, out)
    assert ExTorch.allclose(out, expected)
  end

  test "nonzero/1" do
    input = ExTorch.eye(4)

    expected =
      ExTorch.tensor(
        [
          [0, 0],
          [1, 1],
          [2, 2],
          [3, 3]
        ],
        dtype: :long
      )

    out = ExTorch.nonzero(input)
    assert ExTorch.equal(out, expected)
  end

  test "nonzero/2" do
    input = ExTorch.eye(4)

    expected =
      ExTorch.tensor(
        [
          [0, 0],
          [1, 1],
          [2, 2],
          [3, 3]
        ],
        dtype: :long
      )

    out = ExTorch.empty_like(expected)
    ExTorch.nonzero(input, out)
    assert ExTorch.equal(out, expected)
  end

  test "nonzero/2 with kwargs" do
    input = ExTorch.rand({3, 4}) |> ExTorch.ge(0.2) |> ExTorch.Tensor.to(dtype: :float)
    out_idx = ExTorch.nonzero(input, as_tuple: true)
    assert ExTorch.all(ExTorch.ne(input[out_idx], 0.0))
  end

  test "permute/2" do
    input = ExTorch.rand({3, 4, 5, 2})
    out = ExTorch.permute(input, {2, -1, 0, 1})
    assert out.size == {5, 2, 3, 4}
  end

  test "vstack/1" do
    input = ExTorch.rand({5, 10, 2})
    parts = [input[0..3], input[3..5]]
    out = ExTorch.vstack(parts)
    assert ExTorch.allclose(out, input)
  end

  test "vstack/2" do
    input = ExTorch.rand({5, 10, 2})
    parts = [input[0..3], input[3..5]]
    out = ExTorch.empty_like(input)
    ExTorch.vstack(parts, out)
    assert ExTorch.allclose(out, input)
  end

  test "scatter/4" do
    src = ExTorch.arange(1, 11) |> ExTorch.reshape({2, 5})
    index = ExTorch.tensor([[0, 1, 2, 0]], dtype: :int64)
    input = ExTorch.zeros({3, 5}, dtype: src.dtype)

    expected =
      ExTorch.tensor([
        [1.0000, 0.0000, 0.0000, 4.0000, 0.0000],
        [0.0000, 2.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 3.0000, 0.0000, 0.0000]
      ])

    out = ExTorch.scatter(input, 0, index, src)
    assert ExTorch.allclose(out, expected)
  end

  test "scatter/5" do
    src = ExTorch.arange(1, 11) |> ExTorch.reshape({2, 5})
    index = ExTorch.tensor([[0, 1, 2], [0, 1, 4]], dtype: :int64)
    input = ExTorch.zeros({3, 5}, dtype: src.dtype)

    out = ExTorch.empty_like(input)

    expected =
      ExTorch.tensor([
        [1.0000, 2.0000, 3.0000, 0.0000, 0.0000],
        [6.0000, 7.0000, 0.0000, 0.0000, 8.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
      ])

    ExTorch.scatter(input, 1, index, src, out)
    assert ExTorch.allclose(out, expected)
  end

  test "scatter/5 with kwargs" do
    src = ExTorch.arange(1, 11) |> ExTorch.reshape({2, 5})
    index = ExTorch.tensor([[0, 1, 2], [0, 1, 4]], dtype: :int64)
    input = ExTorch.zeros({3, 5}, dtype: src.dtype)

    expected =
      ExTorch.tensor([
        [1.0000, 2.0000, 3.0000, 0.0000, 0.0000],
        [6.0000, 7.0000, 0.0000, 0.0000, 8.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
      ])

    ExTorch.scatter(input, 1, index, src, inplace: true)
    assert ExTorch.allclose(input, expected)
  end

  test "scatter/6" do
    src = ExTorch.arange(1, 11) |> ExTorch.reshape({2, 5})
    index = ExTorch.tensor([[0, 1, 2, 0]], dtype: :int64)
    input = ExTorch.zeros({3, 5}, dtype: src.dtype)

    expected =
      ExTorch.tensor([
        [1.0000, 0.0000, 0.0000, 4.0000, 0.0000],
        [0.0000, 2.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 3.0000, 0.0000, 0.0000]
      ])

    ExTorch.scatter(input, 0, index, src, nil, true)
    assert ExTorch.allclose(input, expected)
  end

  test "diagonal_scatter/2" do
    input = ExTorch.zeros({3, 3})
    expected = ExTorch.eye(3)
    out = ExTorch.diagonal_scatter(input, ExTorch.ones(3))
    assert ExTorch.allclose(out, expected)
  end

  test "diagonal_scatter/3" do
    input = ExTorch.zeros({3, 3})
    expected = ExTorch.eye(3)
    out = ExTorch.diagonal_scatter(input, ExTorch.ones(3), 0)
    assert ExTorch.allclose(out, expected)
  end

  test "diagonal_scatter/3 with kwargs" do
    input = ExTorch.zeros({3, 3})

    expected =
      ExTorch.tensor([
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000]
      ])

    out = ExTorch.diagonal_scatter(input, ExTorch.ones(2), offset: 1)
    assert ExTorch.allclose(out, expected)
  end

  test "diagonal_scatter/4" do
    input = ExTorch.zeros({3, 3, 3})
    base = ExTorch.eye(3) |> ExTorch.unsqueeze(0)
    expected = ExTorch.cat([base, base, base], 0)

    out = ExTorch.diagonal_scatter(input, ExTorch.ones({3, 3}), 0, 2)
    assert ExTorch.allclose(out, expected)
  end

  test "diagonal_scatter/4 with kwargs" do
    input = ExTorch.zeros({3, 3, 3})

    advanced_index = [
      [0, 0, 0, 1, 1, 1, 2, 2, 2],
      [0, 1, 2, 0, 1, 2, 0, 1, 2],
      [0, 0, 0, 1, 1, 1, 2, 2, 2]
    ]

    expected = ExTorch.index_put(input, advanced_index, 1.0)

    out = ExTorch.diagonal_scatter(input, ExTorch.ones({3, 3}), 0, dim2: -1)
    assert ExTorch.allclose(out, expected)
  end

  test "diagonal_scatter/5" do
    input = ExTorch.zeros({3, 3, 3})
    expected = ExTorch.index_put(input, [[0, 1, 2], [0, 1, 2]], 1.0)

    out = ExTorch.diagonal_scatter(input, ExTorch.ones({3, 3}), 0, 0, 1)
    assert ExTorch.allclose(out, expected)
  end

  test "diagonal_scatter/6" do
    input = ExTorch.zeros({3, 3, 3})
    expected = ExTorch.index_put(input, [[0, 1, 2], [0, 1, 2]], 1.0)

    out = ExTorch.empty_like(input)
    ExTorch.diagonal_scatter(input, ExTorch.ones({3, 3}), 0, 0, 1, out)
    assert ExTorch.allclose(out, expected)
  end

  test "select_scatter/4" do
    input = ExTorch.rand({3, 2, 2})
    src = ExTorch.rand({2, 2})
    expected = ExTorch.index_put(input, 0, src)
    out = ExTorch.select_scatter(input, src, 0, 0)

    assert ExTorch.allclose(out, expected)
  end

  test "select_scatter/5" do
    input = ExTorch.rand({3, 2, 2})
    src = ExTorch.rand({3, 2})
    expected = ExTorch.index_put(input, [:"::", 1], src)
    out = ExTorch.empty_like(input)

    ExTorch.select_scatter(input, src, 1, 1, out)
    assert ExTorch.allclose(out, expected)
  end

  test "slice_scatter/2" do
    input = ExTorch.zeros({4, 5})
    src = ExTorch.rand({4, 5})
    out = ExTorch.slice_scatter(input, src)

    assert ExTorch.allclose(out, src)
  end

  test "slice_scatter/3" do
    input = ExTorch.zeros({4, 5})
    src = ExTorch.rand({4, 5})
    out = ExTorch.slice_scatter(input, src, 1)
    assert ExTorch.allclose(out, src)
  end

  test "slice_scatter/3 with kwargs" do
    input = ExTorch.zeros({8, 8})
    src = ExTorch.rand({8, 3})
    expected = ExTorch.index_put(input, [:"::", 2..5], src)

    out = ExTorch.slice_scatter(input, src, dim: 1, start: 2, stop: 5)
    assert ExTorch.allclose(out, expected)
  end

  test "slice_scatter/4" do
    input = ExTorch.zeros({8, 8})
    src = ExTorch.rand({3, 8})
    expected = ExTorch.index_put(input, [5..8], src)

    out = ExTorch.slice_scatter(input, src, 0, 5)
    assert ExTorch.allclose(out, expected)
  end

  test "slice_scatter/4 with kwargs" do
    input = ExTorch.zeros({8, 8})
    src = ExTorch.rand({8, 3})
    expected = ExTorch.index_put(input, [:"::", 2..7//2], src)

    out = ExTorch.slice_scatter(input, src, 1, 2, stop: 7, step: 2)
    assert ExTorch.allclose(out, expected)
  end

  test "slice_scatter/5" do
    input = ExTorch.zeros({8, 8})
    src = ExTorch.rand({2, 8})
    expected = ExTorch.index_put(input, [4..6], src)

    out = ExTorch.slice_scatter(input, src, 0, 4, 6)
    assert ExTorch.allclose(out, expected)
  end

  test "slice_scatter/5 with kwargs" do
    input = ExTorch.zeros({8, 8})
    src = ExTorch.rand({8, 3})
    expected = ExTorch.index_put(input, [:"::", 0..7//3], src)

    out = ExTorch.slice_scatter(input, src, 1, 0, stop: 7, step: 3)
    assert ExTorch.allclose(out, expected)
  end

  test "slice_scatter/6" do
    input = ExTorch.zeros({8, 8})
    src = ExTorch.rand({2, 8})
    expected = ExTorch.index_put(input, [1..5//3], src)

    out = ExTorch.slice_scatter(input, src, 0, 1, 5, 3)
    assert ExTorch.allclose(out, expected)
  end

  test "test_scatter/6 with kwargs" do
    input = ExTorch.zeros({8, 8})
    src = ExTorch.rand({2, 8})
    expected = ExTorch.index_put(input, [1..5//3], src)
    out = ExTorch.empty_like(input)

    ExTorch.slice_scatter(input, src, 0, 1, 5, 3, out: out)
    assert ExTorch.allclose(out, expected)
  end

  test "test_scatter/7" do
    input = ExTorch.zeros({8, 8})
    src = ExTorch.rand({8, 3})
    expected = ExTorch.index_put(input, [:"::", 0..7//3], src)
    out = ExTorch.empty_like(input)

    ExTorch.slice_scatter(input, src, 1, 0, 7, 3, out)
    assert ExTorch.allclose(out, expected)
  end

  test "scatter_add/4" do
    input = ExTorch.zeros({3, 5})
    index = ExTorch.tensor([[0, 1, 2, 0, 0]], dtype: :int64)
    src = ExTorch.rand({2, 5})

    expected =
      ExTorch.index_put(
        input,
        [[0, 1, 2, 0, 0], [0, 1, 2, 3, 4]],
        src[0]
      )

    out = ExTorch.scatter_add(input, 0, index, src)
    assert ExTorch.allclose(out, expected)
  end

  test "scatter_add/5" do
    input = ExTorch.zeros({3, 5})

    index =
      ExTorch.tensor(
        [
          [0, 1, 2, 0, 0]
        ],
        dtype: :int64
      )

    src = ExTorch.rand({2, 5})

    expected =
      ExTorch.index_put(
        input,
        [[0, 1, 2, 0, 0], [0, 1, 2, 3, 4]],
        src[0]
      )

    out = ExTorch.empty_like(input)
    ExTorch.scatter_add(input, 0, index, src, out)
    assert ExTorch.allclose(out, expected)
  end

  test "scatter_add/5 with kwargs" do
    input = ExTorch.zeros({3, 5})

    index =
      ExTorch.tensor(
        [
          [0, 1, 2, 0, 0]
        ],
        dtype: :int64
      )

    src = ExTorch.rand({2, 5})

    expected =
      ExTorch.index_put(
        input,
        [[0, 1, 2, 0, 0], [0, 1, 2, 3, 4]],
        src[0]
      )

    ExTorch.scatter_add(input, 0, index, src, inplace: true)
    assert ExTorch.allclose(input, expected)
  end

  test "scatter_reduce/5" do
    input = ExTorch.rand({3, 5})

    index =
      ExTorch.tensor(
        [
          [0, 1, 2, 0, 0],
          [0, 1, 2, 2, 2]
        ],
        dtype: :int64
      )

    src = ExTorch.rand({2, 5})
    expected = ExTorch.scatter_add(input, 0, index, src)
    out = ExTorch.scatter_reduce(input, 0, index, src, :sum)
    assert ExTorch.allclose(out, expected)
  end

  test "scatter_reduce/6" do
    input = ExTorch.zeros({3, 5})

    index =
      ExTorch.tensor(
        [
          [0, 1, 2, 0, 0],
          [0, 1, 2, 2, 2]
        ],
        dtype: :int64
      )

    src = ExTorch.rand({2, 5})
    expected = ExTorch.scatter_add(input, 0, index, src)
    out = ExTorch.scatter_reduce(input, 0, index, src, :sum, false)
    assert ExTorch.allclose(out, expected)
  end

  test "scatter_reduce/7 with out" do
    input = ExTorch.rand({3, 5})

    index =
      ExTorch.tensor(
        [
          [0, 1, 2, 0, 0],
          [0, 1, 2, 2, 2]
        ],
        dtype: :int64
      )

    src = ExTorch.rand({2, 5})
    expected = ExTorch.scatter_add(input, 0, index, src)
    out = ExTorch.empty_like(input)
    ExTorch.scatter_reduce(input, 0, index, src, :sum, true, out: out)
    assert ExTorch.allclose(out, expected)
  end

  test "scatter_reduce/7 with inplace" do
    input = ExTorch.rand({3, 5})

    index =
      ExTorch.tensor(
        [
          [0, 1, 2, 0, 0],
          [0, 1, 2, 2, 2]
        ],
        dtype: :int64
      )

    src = ExTorch.rand({2, 5})
    expected = ExTorch.scatter_add(input, 0, index, src)
    ExTorch.scatter_reduce(input, 0, index, src, :sum, true, inplace: true)
    assert ExTorch.allclose(input, expected)
  end

  test "split/2" do
    input = ExTorch.rand({10, 3})
    expected = [input[0..5], input[5..10]]
    out = ExTorch.split(input, 5)

    assert out
           |> Enum.zip(expected)
           |> Enum.reduce(true, fn {o, e}, acc -> ExTorch.allclose(o, e) and acc end)
  end

  test "split/3" do
    input = ExTorch.rand({3, 10})
    expected = [input[{:"::", 0..6}], input[{:"::", 6..10}]]
    out = ExTorch.split(input, [6, 4], 1)
    assert out
           |> Enum.zip(expected)
           |> Enum.reduce(true, fn {o, e}, acc -> ExTorch.allclose(o, e) and acc end)
  end

  test "squeeze/1" do
    input = ExTorch.empty({1, 3, 1, 10, 1})
    expected = {3, 10}
    out = ExTorch.squeeze(input)
    assert out.size == expected
  end

  test "squeeze/2 with a single dimension" do
    input = ExTorch.empty({1, 3, 1, 10, 1})
    expected = {1, 3, 10, 1}
    out = ExTorch.squeeze(input, 2)
    assert out.size == expected
  end

  test "squeeze/2 with a dimension list" do
    input = ExTorch.empty({1, 3, 1, 10, 1})
    expected = {3, 10, 1}
    out = ExTorch.squeeze(input, {0, 2})
    assert out.size == expected
  end

  test "stack/1" do
    input = ExTorch.rand({3, 4, 2})
    exp_input = ExTorch.unsqueeze(input, 0)
    expected = ExTorch.cat([exp_input, exp_input, exp_input])

    out = ExTorch.stack([input, input, input])
    assert ExTorch.allclose(out, expected)
  end

  test "stack/2" do
    input = ExTorch.rand({3, 4, 2})
    exp_input = ExTorch.unsqueeze(input, 2)
    expected = ExTorch.cat([exp_input, exp_input, exp_input], 2)

    out = ExTorch.stack([input, input, input], 2)
    assert ExTorch.allclose(out, expected)
  end

  test "stack/2 with kwargs" do
    input = ExTorch.rand({3, 4, 2})
    exp_input = ExTorch.unsqueeze(input, 3)
    expected = ExTorch.cat([exp_input, exp_input, exp_input], dim: 3)

    out = ExTorch.stack([input, input, input], dim: 3)
    assert ExTorch.allclose(out, expected)
  end

  test "stack/3" do
    input = ExTorch.rand({3, 4, 2})
    exp_input = ExTorch.unsqueeze(input, 1)
    expected = ExTorch.cat([exp_input, exp_input, exp_input], 1)

    out = ExTorch.empty_like(expected)
    ExTorch.stack([input, input, input], 1, out)
    assert ExTorch.allclose(out, expected)
  end

  test "t/1" do
    input = ExTorch.rand({2, 6})
    expected = ExTorch.transpose(input, 0, 1)
    out = ExTorch.t(input)
    assert ExTorch.allclose(out, expected)
  end

  test "take/2" do
    input = ExTorch.rand({3, 3})
    expected = input[{[0, 1, 2], [1, 2, 0]}]
    out = ExTorch.take(input, ExTorch.tensor([1, 5, 6], dtype: :int64))
    assert ExTorch.allclose(out, expected)
  end

  test "take_along_dim/2" do
    input = ExTorch.rand({3, 4})
    expected = ExTorch.max(input)
    indices = ExTorch.argmax(input)

    out = ExTorch.take_along_dim(input, indices)
    assert ExTorch.allclose(out, expected)
  end

  test "take_along_dim/3" do
    input = ExTorch.rand({3, 4})
    {expected, indices} = ExTorch.sort(input, -1)

    out = ExTorch.take_along_dim(input, indices, -1)
    assert ExTorch.allclose(out, expected)
  end

  test "take_along_dim/4" do
    input = ExTorch.rand({3, 4})
    {expected, indices} = ExTorch.min(input, 0, keepdim: true)

    out = ExTorch.empty_like(expected)
    ExTorch.take_along_dim(input, indices, 0, out)
    assert ExTorch.allclose(out, expected)
  end
end
