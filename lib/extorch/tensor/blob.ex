defmodule ExTorch.Tensor.Blob do
  @moduledoc """
  Zero-copy tensor exchange between ExTorch and other tensor frameworks.

  This module provides functions for sharing tensor memory across framework
  boundaries without copying data. It works by exchanging raw memory pointers
  between libraries that share the same process address space (e.g., ExTorch
  and Torchx/Nx, both backed by libtorch).

  ## Safety

  **Memory lifetime is the caller's responsibility.** When you create a tensor
  via `from_blob/2`, the source memory must remain valid for the lifetime of
  the returned tensor. The `%ExTorch.Tensor.Blob{}` struct returned by
  `to_blob/1` holds a reference to the source tensor, preventing it from
  being garbage collected.

  ## Example: ExTorch → Nx (via Torchx)

      blob = ExTorch.Tensor.Blob.to_blob(extorch_tensor)
      # Pass blob.ptr, blob.shape, blob.strides, blob.dtype to Torchx

  ## Example: Nx (via Torchx) → ExTorch

      view = ExTorch.Tensor.Blob.from_blob(
        %{ptr: torchx_data_ptr, shape: {3, 4}, dtype: :float32},
        owner: nx_tensor  # prevents GC of source
      )
      view.tensor  # the ExTorch tensor (shares memory with nx_tensor)
  """

  @type t :: %__MODULE__{
          ptr: integer(),
          shape: tuple(),
          strides: [integer()],
          dtype: ExTorch.DType.dtype(),
          device: ExTorch.Device.device(),
          element_size: integer(),
          owner: ExTorch.Tensor.t()
        }

  defstruct [:ptr, :shape, :strides, :dtype, :device, :element_size, :owner]

  @typedoc """
  A tensor view backed by foreign memory. Holds a reference to both
  the tensor and the memory owner, preventing the owner from being GC'd.
  """
  @type view :: %ExTorch.Tensor.BlobView{
          tensor: ExTorch.Tensor.t(),
          owner: term()
        }

  @doc """
  Extract raw memory information from an ExTorch tensor.

  Returns a `%ExTorch.Tensor.Blob{}` struct containing the pointer and metadata.
  The struct holds a reference to the source tensor, keeping it alive.

  ## Arguments
    - `tensor` - The source `%ExTorch.Tensor{}`.

  ## Returns
  A `%ExTorch.Tensor.Blob{}` with fields:
    - `ptr` - Raw memory address (integer). Valid for CPU and CUDA pointers.
    - `shape` - Tensor shape as a tuple.
    - `strides` - Stride per dimension as a list of integers.
    - `dtype` - Element data type.
    - `device` - Device the memory lives on.
    - `element_size` - Size of each element in bytes.
    - `owner` - Reference to the source tensor (prevents GC).
  """
  @spec to_blob(ExTorch.Tensor.t()) :: t()
  def to_blob(%ExTorch.Tensor{} = tensor) do
    %__MODULE__{
      ptr: ExTorch.Native.data_ptr(tensor),
      shape: tensor.size,
      strides: ExTorch.Native.strides(tensor),
      dtype: tensor.dtype,
      device: tensor.device,
      element_size: ExTorch.Native.element_size(tensor),
      owner: tensor
    }
  end

  @doc """
  Create an ExTorch tensor from a raw memory pointer (zero-copy).

  The returned tensor shares the memory at the given pointer. **The caller
  must ensure the source memory outlives the returned tensor.**

  Pass the `:owner` option to hold a reference to the source object,
  preventing it from being garbage collected.

  ## Arguments
    - `blob_info` - A map or struct with keys: `:ptr`, `:shape`, `:dtype`.
      Optional keys: `:strides`, `:device`.
    - `opts` - Keyword options:
      - `:owner` - Any term to keep alive alongside the tensor (e.g., the
        source Nx tensor or Torchx reference). Stored in the returned struct.

  ## Returns
  An `%ExTorch.Tensor{}`. If `:owner` is given, the tensor's internal
  reference ensures the owner is not garbage collected.
  """
  @doc """
  Create an ExTorch tensor from a raw memory pointer (zero-copy).

  The returned tensor shares the memory at the given pointer. **The caller
  must ensure the source memory outlives the returned tensor.**

  When `:owner` is provided, returns an `%ExTorch.Tensor.BlobView{}` that
  holds references to both the tensor and the owner, preventing the owner
  from being garbage collected while the view is alive.

  ## Arguments
    - `blob_info` - A map, `%Blob{}`, or struct with keys: `:ptr`, `:shape`, `:dtype`.
      Optional keys: `:strides`, `:device`.
    - `opts` - Keyword options:
      - `:owner` - Any term to keep alive alongside the tensor (e.g., the
        source Nx tensor or Torchx reference).

  ## Returns
  An `%ExTorch.Tensor.BlobView{}` (when `:owner` is given) or
  `%ExTorch.Tensor{}` (without `:owner` -- caller assumes lifetime responsibility).
  """
  @spec from_blob(map(), keyword()) :: view() | ExTorch.Tensor.t()
  def from_blob(blob_info, opts \\ []) do
    ptr = Map.fetch!(blob_info, :ptr)
    shape = Map.fetch!(blob_info, :shape)
    dtype = Map.fetch!(blob_info, :dtype)
    device = Map.get(blob_info, :device, :cpu)

    strides =
      case Map.get(blob_info, :strides) do
        nil -> default_strides(shape, dtype)
        s when is_list(s) -> List.to_tuple(s)
        s when is_tuple(s) -> s
      end

    shape = if is_list(shape), do: List.to_tuple(shape), else: shape

    tensor = ExTorch.Native.from_blob(ptr, shape, strides, dtype, device)

    case Keyword.get(opts, :owner) do
      nil ->
        tensor

      owner ->
        %ExTorch.Tensor.BlobView{tensor: tensor, owner: owner}
    end
  end

  # Compute default C-contiguous strides from shape
  defp default_strides(shape, _dtype) do
    shape = if is_tuple(shape), do: Tuple.to_list(shape), else: shape

    shape
    |> Enum.reverse()
    |> Enum.reduce({[], 1}, fn dim, {strides, acc} ->
      {[acc | strides], acc * dim}
    end)
    |> elem(0)
    |> List.to_tuple()
  end
end
