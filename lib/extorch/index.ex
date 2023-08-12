defmodule ExTorch.Index do
  @moduledoc """
  An index is an object that can act as an accessor to a ``ExTorch.Tensor``.
  ExTorch has five kinds of indices:
  """

  @typedoc """
  Dimension appending/removing indices.

  ## Description
  - `nil` - Adds an empty dimension on the specified position.
  - `true` - Same as `nil`
  - `false` - Adds a zero dimension on the specified position.
  """
  @type dim_modifier_index :: nil | boolean()

  @typedoc """
  Access a slice of elements in the tensor given a range.

  ## Description
  - `start..end//step` - A standard Elixir range struct, it will be non-inclusive
     on `end`, as opposed to its usual behaviour in Elixir.
  - `Slice.t()` - See `ExTorch.Index.Slice`
  - `:::` - Same as invoking `ExTorch.slice/0`.
  """
  @type range_index :: Range.t() | ExTorch.Index.Slice.t() | :"::"

  @typedoc """
  Access a tensor given a particular integer index.
  """
  @type integer_index :: integer()

  @type integer_list :: integer_index() | [integer_list()]

  @typedoc """
  Access a tensor given another tensor or a list of integers.
  """
  @type tensor_list_index :: ExTorch.Tensor.t() | integer_list()

  @typedoc """
  Skip dimensions in between other ones.

  # Notes
  Ellipsis indices will behave as a sequence of empty
  `ExTorch.Index.Slice`s between the dimensions that are specified.
  e.g.,

  ```elixir
  a = ExTorch.empty({3, 4, 5, 6})

  # Indexing will yield a tensor of size {4, 5}
  indexing = ExTorch.index(a, [0, :..., 1])
  ```
  """
  @type ellipsis_index :: :ellipsis | :...

  @typedoc """
  An individual index element.
  """
  @type index :: dim_modifier_index() | range_index() | tensor_list_index() | ellipsis_index() | integer_index()

  @typedoc """
  An actual index element, after aliases are interpreted.
  """
  @type actual_index :: nil | boolean() | ExTorch.Index.Slice.t() | integer() | ExTorch.Tensor.t() | :ellipsis

  @typedoc """
  A complete possible index. It can be either a puntual index or a list of puntual indices.
  """
  @type t :: index() | [index()] | tuple()
end
