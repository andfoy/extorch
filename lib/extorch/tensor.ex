defmodule ExTorch.Tensor do
  @moduledoc """
  An ``ExTorch.Tensor`` is a multi-dimensional matrix containing elements of a single data type.
  """
  @behaviour Access

  use ExTorch.DelegateWithDocs
  import ExTorch.ModuleMixin

  @typedoc """
  An ``ExTorch.Tensor`` is a multi-dimensional matrix containing elements of a single data type.
  """
  @type t :: %__MODULE__{
          resource: any(),
          reference: reference(),
          size: tuple(),
          dtype: ExTorch.DType.dtype(),
          device: ExTorch.Device.device()
        }

  defstruct [
    # The actual Tensor NIF resource
    resource: nil,

    # Normally the compiler will happily do stuff like inlining the
    # resource in attributes. This will convert the resource into an
    # empty binary with no warning. This will make that harder to
    # accidentaly do.
    # It also serves as a handy way to tell tensor handles apart.
    reference: nil,

    # Size of the tensor reference
    size: {},

    # Type of the tensor reference
    dtype: :int,

    # Device where the tensor lives on
    device: :cpu
  ]

  @doc false
  def wrap_tensor_ref(resource) do
    %__MODULE__{
      resource: resource,
      reference: make_ref(),
      size: ExTorch.Native.size(resource),
      dtype: ExTorch.Native.dtype(resource),
      device: ExTorch.Native.device(resource)
    }
  end

  defimpl Inspect, for: ExTorch.Tensor do
    import Inspect.Algebra

    def inspect(tensor, opts) do
      # %ExTorch.Tensor{resource: resource} = tensor
      repr = ExTorch.Native.repr(tensor, ExTorch.Utils.PrintOptions.default())

      concat([
        Inspect.Algebra.color("#Tensor<", :string, opts),
        Inspect.Algebra.break(),
        repr,
        Inspect.Algebra.break(),
        to_doc(
          [
            size: tensor.size,
            dtype: tensor.dtype,
            device: tensor.device,
            requires_grad: ExTorch.Native.requires_grad(tensor)
          ],
          opts
        ),
        Inspect.Algebra.color(">", :string, opts)
      ])
    end
  end

  @spec fetch(ExTorch.Tensor.t(), ExTorch.Index.t()) :: {:ok, ExTorch.Tensor.t()}
  @doc """
  Index a tensor using an accessor object. It acts as a alias for `ExTorch.index/2`.
  """
  def fetch(tensor, index) do
    {:ok, ExTorch.index(tensor, index)}
  end

  @doc false
  def pop(_, _) do
    {:error, :not_implemented}
  end

  @doc false
  def get_and_update(_, _, _) do
    {:error, :not_implemented}
  end

  @doc """
  Take a `ExTorch.Tensor` or `ExTorch.Scalar` definition and convert it into
  a `ExTorch.Tensor`.

  ## Arguments
  - `input` - the input to convert into an `ExTorch.Tensor`.
  - `device` - an optional device to map the tensor into. This will only take effect
  when `input` is one of any `ExTorch.Scalar` definitions. Default: `:cpu`
  """
  @spec scalar_to_tensor(
          ExTorch.Tensor.t() | ExTorch.Scalar.scalar_or_list(),
          ExTorch.Device.device()
        ) :: ExTorch.Tensor.t()
  def scalar_to_tensor(input, device \\ :cpu) do
    case input do
      %ExTorch.Tensor{} ->
        input

      _ ->
        ExTorch.tensor(input,
          device: device,
          requires_grad: false
        )
    end
  end

  extends(ExTorch.Native.Tensor.Info)
end
