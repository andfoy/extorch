defmodule ExTorch.Complex do
  @moduledoc """
  An ``ExTorch.Complex`` is a struct that represents a complex number with real and imaginary parts.

  **Note**: This struct only provides a mechanism to exchange imaginary value between Elixir and LibTorch and therefore
  does not provide any arithmetic functions.
  """

  @typedoc """
  An ``ExTorch.Complex`` is a struct that represents a complex number with real and imaginary parts.
  """
  @type t :: %__MODULE__{
    real: number(),
    imaginary: number()
  }

  defstruct [
    real: 0,
    imaginary: 0
  ]

  defimpl Inspect, for: ExTorch.Complex do
    import Inspect.Algebra

    def inspect(%ExTorch.Complex{real: re, imaginary: im}, _opts) do
      {sign, im} =
        case im >= 0 do
          true -> {"+", im}
          false -> {"-", Kernel.abs(im)}
        end
      concat(["#{re}", " ", sign, " ", "#{im}j"])
    end
  end

  defimpl String.Chars, for: ExTorch.Tensor do
    def to_string(%ExTorch.Complex{real: re, imaginary: im}) do
      {sign, im} =
        case im >= 0 do
          true -> {"+", im}
          false -> {"-", Kernel.abs(im)}
        end
      "#{re} #{sign} #{im}j"
    end
  end

  @spec complex(number(), number()) :: ExTorch.Complex.t()
  @doc """
  Create an `ExTorch.Complex` struct with real and imaginary parts.

  ## Arguments
  - `re` - The real part of the complex number (`number()`)
  - `im` - The imaginary part of the complex number (`number()`)

  ## Returns
  - `complex` - An imaginary number with real part `re` and imaginary part `im` (`ExTorch.Complex`)
  """
  def complex(re, im) do
    %ExTorch.Complex {
      real: re / 1.0,
      imaginary: im / 1.0
    }
  end

  @spec re(ExTorch.Complex.t()) :: number()
  @doc """
  Retrieve the real part of a complex number

  ## Arguments
  - `complex` - A complex number (`ExTorch.Complex`)

  ## Returns
  - `re` - The real part of the input complex number. (`number()`)
  """
  def re(%ExTorch.Complex{real: real}) do
    real
  end

  @spec im(ExTorch.Complex.t()) :: number()
  @doc """
  Retrieve the imaginary part of a complex number

  ## Arguments
  - `complex` - A complex number (`ExTorch.Complex`)

  ## Returns
  - `im` - The imaginary part of the input complex number. (`number()`)
  """
  def im(%ExTorch.Complex{imaginary: im}) do
    im
  end

end
