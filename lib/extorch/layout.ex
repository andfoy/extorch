defmodule ExTorch.Layout do
  @moduledoc """
  A ``torch.layout`` is an object that represents the memory layout of a ``torch.Tensor``.
  Currently, we support ``torch.strided`` (dense Tensors) and have beta support for
  ``torch.sparse_coo`` (sparse COO Tensors).

  ``torch.strided`` represents dense Tensors and is the memory layout that is
  most commonly used. Each strided tensor has an associated ``torch.Storage``,
  which holds its data. These tensors provide multi-dimensional, strided view of a storage.
  Strides are a list of integers: the k-th stride represents the jump in the memory
  necessary to go from one element to the next one in the k-th dimension of the Tensor.
  This concept makes it possible to perform many tensor operations efficiently.
  """

  @typedoc """
  A ``torch.layout`` is an object that represents the memory layout of a ``torch.Tensor``.
  `nil` is only used when copying layouts between tensors.
  """
  @type layout :: :strided | :sparse | nil

  @layout [:strided, :sparse]

  defguard is_layout(layout) when layout in @layout
end
