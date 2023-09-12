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
          real: number() | :nan | :inf | :ninf,
          imaginary: number() | :nan | :inf | :ninf
        }

  defstruct real: 0,
            imaginary: 0

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
        case im do
          :nan -> {"+", "nan"}
          :inf -> {"+", "inf"}
          :ninf -> {"-", "inf"}
          im when im >= 0 -> {"+", im}
          im -> {"-", Kernel.abs(im)}
        end

      re =
        case re do
          :nan -> "nan"
          :inf -> "inf"
          :ninf -> "-inf"
          r -> r
        end

      "#{re} #{sign} #{im}j"
    end
  end

  @spec complex(number() | :nan | :inf | :ninf, number() | :nan | :inf | :ninf) ::
          ExTorch.Complex.t()
  @doc """
  Create an `ExTorch.Complex` struct with real and imaginary parts.

  ## Arguments
  - `re` - The real part of the complex number (`number() | :nan | :inf | :ninf`)
  - `im` - The imaginary part of the complex number (`number() | :nan | :inf | :ninf`)

  ## Returns
  - `complex` - An imaginary number with real part `re` and imaginary part `im` (`ExTorch.Complex`)
  """
  def complex(re, im) do
    re = case re do
      r when is_number(r) -> r / 1.0
      _ -> re
    end

    im = case im do
      i when is_number(i) -> i / 1.0
      _ -> im
    end

    %ExTorch.Complex{
      real: re,
      imaginary: im
    }
  end

  @spec re(ExTorch.Complex.t()) :: number() | :nan | :inf | :ninf
  @doc """
  Retrieve the real part of a complex number

  ## Arguments
  - `complex` - A complex number (`ExTorch.Complex`)

  ## Returns
  - `re` - The real part of the input complex number. (`number() | :nan | :inf | :ninf`)
  """
  def re(%ExTorch.Complex{real: real}) do
    real
  end

  @spec im(ExTorch.Complex.t()) :: number() | :nan | :inf | :ninf
  @doc """
  Retrieve the imaginary part of a complex number

  ## Arguments
  - `complex` - A complex number (`ExTorch.Complex`)

  ## Returns
  - `im` - The imaginary part of the input complex number. (`number() | :nan | :inf | :ninf`)
  """
  def im(%ExTorch.Complex{imaginary: im}) do
    im
  end
end
