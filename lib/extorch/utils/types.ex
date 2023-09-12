defmodule ExTorch.Utils.Types do
  @moduledoc """
  General type hierarchy comparison utils
  """

  @doc """
  Given two basic types, compare them and return the type that subsumes the other one.
  """
  @spec compare_types(ExTorch.DType.base_type(), ExTorch.DType.base_type()) ::
          ExTorch.DType.base_type()
  def compare_types(:bool, y) do
    y
  end

  def compare_types(x, :bool) do
    x
  end

  def compare_types(:uint8, y) when y in [:int32, :int64, :float32, :float64] do
    y
  end

  def compare_types(x, :uint8) when x in [:int32, :int64, :float32, :float64] do
    x
  end

  def compare_types(:int32, y) when y in [:float32, :float64] do
    :float64
  end

  def compare_types(x, :int32) when x in [:float32, :float64] do
    :float64
  end

  def compare_types(:int32, y) do
    y
  end

  def compare_types(x, :int32) do
    x
  end

  def compare_types(:int64, y) when y in [:float32, :float64] do
    :float64
  end

  def compare_types(x, :int64) when x in [:float32, :float64] do
    :float64
  end

  def compare_types(:float32, y) do
    y
  end

  def compare_types(x, :float32) do
    x
  end

  def compare_types(:complex64, y) when y not in [:complex128] do
    :complex64
  end

  def compare_types(x, :complex64) when x not in [:complex128] do
    :complex64
  end

  def compare_types(:complex128, _) do
    :complex128
  end

  def compare_types(_, :complex128) do
    :complex128
  end

  def compare_types(t, t) do
    t
  end

  @doc """
  Given a list/tuples with elements or lists/tuples of elements, determine the base type
  `ExTorch.DType.base_type()` that the list should have when converting it into a tensor.
  """
  @spec collect_types(list() | tuple() | number() | boolean(), MapSet.t()) :: MapSet.t()
  def collect_types([], acc) do
    acc
  end

  def collect_types([h | t], acc) do
    head_types = collect_types(h, MapSet.new())
    collect_types(t, MapSet.union(head_types, acc))
  end

  def collect_types(tup, acc) when is_tuple(tup) do
    collect_types(Tuple.to_list(tup), acc)
  end

  def collect_types(integer, acc) when is_integer(integer) and integer < 256 and integer >= 0 do
    MapSet.put(acc, :uint8)
  end

  def collect_types(integer, acc) when is_integer(integer) and integer <= 2_147_483_647 do
    MapSet.put(acc, :int32)
  end

  def collect_types(integer, acc) when is_integer(integer) do
    MapSet.put(acc, :int64)
  end

  def collect_types(bool, acc) when is_boolean(bool) do
    MapSet.put(acc, :bool)
  end

  def collect_types(float, acc) when is_float(float) and float <= 3.4_028_235e38 do
    dtype =
      case ExTorch.get_default_dtype() do
        :float32 -> :float32
        :float64 -> :float64
        _ -> :float32
      end

    MapSet.put(acc, dtype)
  end

  def collect_types(float, acc) when is_float(float) do
    MapSet.put(acc, :float64)
  end

  def collect_types(%ExTorch.Complex{real: re, imaginary: im}, acc) do
    default_dtype =
      case ExTorch.get_default_dtype() do
        :float32 -> :complex64
        :float64 -> :complex128
        _ -> :complex64
      end

    choose_complex_dtype = fn
      :complex64, :complex64 -> :complex64
      _, :complex128 -> :complex128
    end

    re_dtype =
      case re <= 3.4_028_235e38 do
        true -> choose_complex_dtype.(:complex64, default_dtype)
        false -> :complex128
      end

    im_dtype =
      case im <= 3.4_028_235e38 do
        true -> choose_complex_dtype.(:complex64, default_dtype)
        false -> :complex128
      end

    MapSet.put(acc, compare_types(re_dtype, im_dtype))
  end

  def collect_types(atom_value, acc) when atom_value in [:nan, :inf, :ninf] do
    dtype =
      case ExTorch.get_default_dtype() do
        :float32 -> :float32
        :float64 -> :float64
        _ -> :float32
      end

    MapSet.put(acc, dtype)
  end
end
