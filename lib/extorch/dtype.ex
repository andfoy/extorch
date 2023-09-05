defmodule ExTorch.DType do
  @moduledoc """
  A ``torch.dtype`` is an object that represents the data type of a ``torch.Tensor``.
  ExTorch has twelve different data types:
  """

  @typedoc """
  Integral tensor types, unsigned and signed.
  """
  @type integral_type :: :uint8 | :int8 | :int16 | :int32 | :int64

  @typedoc """
  Floating point tensor types.
  """
  @type floating_type :: :float16 | :bfloat16 | :float32 | :float64

  @typedoc """
  Complex number tensor types.
  """
  @type complex_type :: :complex32 | :complex64 | :complex128

  @typedoc """
  Numeric tensor types
  """
  @type numeric_type :: integral_type() | floating_type()

  @typedoc """
  Basic tensor types
  """
  @type base_type :: :bool | numeric_type() | complex_type()

  @typedoc """
  Quantized floating point tensor types.
  """
  @type quantized_type :: :quint8 | :quint2x4 | :quint4x2 | :qint8 | :qint32

  @typedoc """
  Alias names to other integral/floating tensor types.
  """
  @type alias_type ::
          :byte
          | :char
          | :short
          | :int
          | :long
          | :half
          | :float
          | :double
          | :chalf
          | :cfloat
          | :cdouble
          | :complex_float
          | :complex_double

  @typedoc """
  A torch.dtype is an object that represents the data type of a torch.Tensor.
  """
  @type dtype :: base_type() | complex_type() | quantized_type() | alias_type() | nil

  @dtypes [
    :uint8,
    :int8,
    :int16,
    :int32,
    :int64,
    :float16,
    :bfloat16,
    :float32,
    :float64,
    :bool,
    :complex32,
    :complex64,
    :complex128,
    :quint8,
    :quint2x4,
    :quint4x2,
    :qint8,
    :qint32,
    :byte,
    :char,
    :short,
    :int,
    :long,
    :half,
    :float,
    :double,
    :chalf,
    :cfloat,
    :cdouble
  ]

  defguard is_dtype(dtype) when dtype in @dtypes
end
