defmodule ExTorch.Utils.PrintOptions do
  @moduledoc """
  Tensor printing options.
  """

  @after_compile __MODULE__
  def __after_compile__(_env, bytecode) do
    __MODULE__
    |> :code.which()
    |> to_string()
    |> File.write(bytecode)
  end

  @typedoc """
  An ``ExTorch.Utils.PrintOptions`` is a struct that is used to set different
  printing options to display an ``ExTorch.Tensor`` struct.

  ## Fields
  - `precision`: Number of digits of precision for floating point output. Default: 4

  - `threshold`: Total number of array elements which trigger summarization
    rather than full `repr`. Default: 1000.

  - `edgeitems`: Number of array items in summary at beginning and end of
    each dimension. Default: 3.

  - `linewidth`: The number of characters per line for the purpose of
    inserting line breaks (default = 80). Thresholded matrices will
    ignore this parameter.

  - `sci_mode`: Enable (`true`) or disable (`false`) scientific notation. If
    `nil` (default) is specified, the value is defined by
    the formatter. This value is automatically chosen by the framework.
  """
  @type t :: %__MODULE__{
          precision: integer(),
          threshold: float(),
          edgeitems: integer(),
          linewidth: integer(),
          sci_mode: boolean() | nil
        }

  @derive [ExTorch.Protocol.DefaultStruct]
  defstruct precision: 4,
            threshold: 1000.0,
            edgeitems: 3,
            linewidth: 80,
            sci_mode: nil

  @doc """
  Retrieve the default print options.
  """
  @spec default() :: ExTorch.Utils.PrintOptions.t()
  def default() do
    %__MODULE__{}
  end

  @doc """
  Retrieve the short print options.
  """
  @spec short() :: ExTorch.Utils.PrintOptions.t()
  def short() do
    %__MODULE__{
      precision: 2,
      edgeitems: 2
    }
  end

  @doc """
  Retrieve the full print options.
  """
  @spec full() :: ExTorch.Utils.PrintOptions.t()
  def full() do
    %__MODULE__{
      precision: 4,
      threshold: 1.7e308
    }
  end
end
