defmodule ExTorch.Utils do
  alias ExTorch.Utils.Types, as: Types

  defmodule ListWrapper do
    @typedoc """
    This struct wraps a list of elements or a list with lists of elements into
    a representation suitable to be converted into an ExTorch.Tensor
    """
    @type t :: %__MODULE__{
      list: [number()] | [boolean()],
      size: [integer()],
      dtype: ExTorch.DType.base_type()
    }

    @moduledoc """
    Struct used to represent a list with elements or lists of elements.
    """
    defstruct list: [],
              size: [],
              dtype: nil

  end

  @doc """
  Given a list of elements or a list with lists with elements, this function
  returns a ExTorch.Utils.ListWrapper structure.
  """
  @spec to_list_wrapper(list() | number()) :: __MODULE__.ListWrapper.t()
  def to_list_wrapper([]) do
    {%__MODULE__.ListWrapper{
       list: [],
       size: [],
       dtype: :float32
     }, :float32}
  end

  def to_list_wrapper([_ | _] = input) do
    all_types = Types.collect_types(input, MapSet.new())
    input_size = size(input)

    coerce_type =
      all_types
      |> Enum.to_list()
      |> Enum.reduce(&Types.compare_types/2)

    new_list = convert_list(input, coerce_type, [])
    new_list = flat_list(new_list)
    # new_list = Enum.flat_map(new_list, fn x -> x end)

    {%__MODULE__.ListWrapper{
       list: new_list,
       size: input_size,
       dtype: coerce_type
     }, coerce_type}
  end

  def to_list_wrapper(input) when is_tuple(input) do
    to_list_wrapper(Tuple.to_list(input))
  end

  def to_list_wrapper(input) do
    to_list_wrapper([input])
  end

  defp size(x) do
    size(x, [0])
  end

  defp size([_ | _] = x, []) do
    size(x, [0])
  end

  defp size([h | t], [size | rest]) when is_list(h) do
    head_size = size(h, [0])

    case rest do
      [] -> size(t, [size + 1 | head_size])
      ^head_size -> size(t, [size + 1 | rest])
      _ -> {:error, :size_mismatch}
    end
  end

  defp size([h | t], size) when is_tuple(h) do
    size([Tuple.to_list(h) | t], size)
  end

  defp size([_h | t], [size]) do
    size(t, [size + 1])
  end

  defp size([], size) do
    size
  end

  defp size(tup, size) when is_tuple(tup) do
    size(Tuple.to_list(tup), size)
  end

  defp size(_x, _) do
    []
  end

  defp convert_list([], _type, acc) do
    Enum.reverse(acc)
  end

  defp convert_list([h | t], type, acc) do
    head_converted = convert_list(h, type, [])
    convert_list(t, type, [head_converted | acc])
  end

  defp convert_list(tup, type, acc) when is_tuple(tup) do
    convert_list(Tuple.to_list(tup), type, acc)
  end

  defp convert_list(bool, type, _) when is_boolean(bool) and type in [:float32, :float64] do
    case bool do
      true -> 1.0
      false -> 0.0
    end
  end

  defp convert_list(bool, type, _) when is_boolean(bool) and type in [:uint8, :int32, :int64] do
    case bool do
      true -> 1
      false -> 0
    end
  end

  defp convert_list(integer, type, _) when is_integer(integer) and type in [:float32, :float64] do
    integer / 1
  end

  defp convert_list(value, _, _) do
    value
  end

  defp flat_list(list) do
    flat = flat_list(list, [])
    Enum.reverse(flat)
  end

  defp flat_list([], acc) do
    acc
  end

  defp flat_list([h | t], acc) when is_list(h) do
    flat_list(t, flat_list(h, acc))
  end

  defp flat_list([h | t], acc) do
    flat_list(t, [h|acc])
  end
end
