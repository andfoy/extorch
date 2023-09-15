defmodule ExTorch.Scalar do
  @moduledoc """
  An `ExTorch.Scalar` is any singular value that can be stored in a `ExTorch.Tensor`.
  """

  @typedoc """
  A scalar consists of any value that can be stored in an `ExTorch.Tensor` struct.

  It can be:
  - Any number, floating or integer.
  - The special floating point values :nan, :inf and :ninf, which represent
  the Not-A-Number, positive infinity and negative infinity, respectively.
  - A complex value represented by the `ExTorch.Complex` struct.
  """
  @type t :: number() | :nan | :inf | :ninf | ExTorch.Complex.t()

  @typedoc """
  An scalar list is a list that either contains scalars or scalar lists.
  """
  @type scalar_list :: [t() | [scalar_list()]]

  @typedoc """
  Scalar or scalar list type specifier.
  """
  @type scalar_or_list :: t() | scalar_list() | tuple()

end
