defmodule ExTorch.JIT do
  @moduledoc """
  TorchScript model loading, inference, and management.

  This module provides functions to load pre-trained TorchScript (`.pt`) models,
  run inference, inspect model structure, and manage model lifecycle.

  ## Example

      model = ExTorch.JIT.load("model.pt")
      input = ExTorch.randn({1, 3, 224, 224})
      output = ExTorch.JIT.forward(model, [input])

  """

  alias ExTorch.JIT.Model

  @doc """
  Load a TorchScript model from a file.

  ## Arguments
    - `path`: Path to the `.pt` file.
    - `opts`: Keyword list of options.
      - `:device` - Device to load the model onto (default: `:cpu`).

  ## Returns
  A `%ExTorch.JIT.Model{}` struct.

  ## Examples

      model = ExTorch.JIT.load("model.pt")
      model = ExTorch.JIT.load("model.pt", device: {:cuda, 0})
  """
  @spec load(String.t(), keyword()) :: Model.t()
  def load(path, opts \\ []) do
    device = Keyword.get(opts, :device, :cpu)
    ExTorch.Native.jit_load(path, device)
  end

  @doc """
  Save a TorchScript model to a file.

  ## Arguments
    - `model`: The model to save.
    - `path`: Path to save the `.pt` file.
  """
  @spec save(Model.t(), String.t()) :: :ok
  def save(%Model{} = model, path) do
    ExTorch.Native.jit_save(model, path)
    :ok
  end

  @doc """
  Run the forward method on a model.

  ## Arguments
    - `model`: The loaded model.
    - `inputs`: A list of input tensors.

  ## Returns
  The model output, which can be a tensor, tuple, list, map, or scalar
  depending on the model's return type.

  ## Examples

      input = ExTorch.randn({1, 784})
      output = ExTorch.JIT.forward(model, [input])
  """
  @spec forward(Model.t(), [ExTorch.Tensor.t()]) :: term()
  def forward(%Model{} = model, inputs) when is_list(inputs) do
    ExTorch.Native.jit_forward(model, inputs)
  end

  @doc """
  Invoke a named method on a model.

  ## Arguments
    - `model`: The loaded model.
    - `method_name`: Name of the method to invoke.
    - `inputs`: A list of input tensors.

  ## Returns
  The method output.
  """
  @spec invoke(Model.t(), String.t(), [ExTorch.Tensor.t()]) :: term()
  def invoke(%Model{} = model, method_name, inputs \\ []) when is_list(inputs) do
    ExTorch.Native.jit_invoke_method(model, method_name, inputs)
  end

  @doc """
  Get the names of all methods on a model.

  ## Returns
  A list of method name strings.
  """
  @spec methods(Model.t()) :: [String.t()]
  def methods(%Model{} = model) do
    ExTorch.Native.jit_get_method_names(model)
  end

  @doc """
  Get named parameters of a model.

  ## Returns
  A list of `{name, tensor}` tuples.
  """
  @spec parameters(Model.t()) :: [{String.t(), ExTorch.Tensor.t()}]
  def parameters(%Model{} = model) do
    ExTorch.Native.jit_named_parameters(model)
  end

  @doc """
  Get named buffers of a model.

  ## Returns
  A list of `{name, tensor}` tuples.
  """
  @spec buffers(Model.t()) :: [{String.t(), ExTorch.Tensor.t()}]
  def buffers(%Model{} = model) do
    ExTorch.Native.jit_named_buffers(model)
  end

  @doc """
  Get names of submodules of a model.

  ## Returns
  A list of submodule name strings.
  """
  @spec modules(Model.t()) :: [String.t()]
  def modules(%Model{} = model) do
    ExTorch.Native.jit_named_modules(model)
  end

  @doc """
  Set a model to evaluation mode.
  """
  @spec eval(Model.t()) :: :ok
  def eval(%Model{} = model) do
    ExTorch.Native.jit_set_eval(model)
    :ok
  end

  @doc """
  Set a model to training mode.
  """
  @spec train(Model.t()) :: :ok
  def train(%Model{} = model) do
    ExTorch.Native.jit_set_train(model)
    :ok
  end

  @doc """
  Move a model to a different device.

  ## Arguments
    - `model`: The model to move.
    - `device`: Target device (e.g., `:cpu`, `{:cuda, 0}`).

  ## Returns
  A new `%ExTorch.JIT.Model{}` on the target device.
  """
  @spec to(Model.t(), ExTorch.Device.device()) :: Model.t()
  def to(%Model{} = model, device) do
    ExTorch.Native.jit_to_device(model, device)
  end
end
