defmodule ExTorch.Tensor do
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
    device: :cpu,
  ]

  def wrap_tensor_ref(resource) do
    %__MODULE__{
      resource: resource,
      reference: make_ref(),
      size: ExTorch.Native.size(resource),
      dtype: ExTorch.Native.dtype(resource),
      device: ExTorch.Native.device(resource)
    }
  end
end
