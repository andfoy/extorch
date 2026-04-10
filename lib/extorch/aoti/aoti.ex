defmodule ExTorch.AOTI do
  @moduledoc """
  Load and run inference on AOTInductor-compiled models (`.pt2` packages).

  AOTI (Ahead-of-Time Inductor) models are compiled through `torch.export` and
  `torch._inductor.aoti_compile_and_package()`, producing optimized `.pt2`
  packages with fused kernels. These models offer better inference throughput
  than TorchScript but do not support introspection or weight extraction.

  ## Python export workflow

      import torch
      from torch._inductor import aoti_compile_and_package

      model = MyModel()
      model.eval()
      example_input = torch.randn(1, 10)

      exported = torch.export.export(model, (example_input,))
      aoti_compile_and_package(exported, package_path="model.pt2")

  ## Elixir inference

      model = ExTorch.AOTI.load("model.pt2")
      input = ExTorch.randn({1, 10})
      [output] = ExTorch.AOTI.forward(model, [input])

  ## Availability

  AOTI support requires a libtorch build that includes the inductor runtime.
  Check with `ExTorch.AOTI.available?/0`.
  """

  alias ExTorch.AOTI.Model

  @doc """
  Check if AOTI support is available in the current libtorch build.
  """
  @spec available?() :: boolean()
  def available? do
    ExTorch.Native.aoti_is_available()
  end

  @doc """
  Load an AOTI-compiled model from a `.pt2` package.

  ## Args
    * `path` (`String`) - path to the `.pt2` file.
    * `opts` (`keyword`) - optional arguments:
      * `:model_name` (`String`) - name of the model within the package. Default: `"model"`.
      * `:device_index` (`integer`) - device index for CUDA. Default: `-1` (CPU).

  ## Returns
  An `%ExTorch.AOTI.Model{}` struct.
  """
  @spec load(String.t(), keyword()) :: Model.t()
  def load(path, opts \\ []) do
    ensure_libtorch_loaded()
    model_name = Keyword.get(opts, :model_name, "model")
    device_index = Keyword.get(opts, :device_index, -1)
    ExTorch.Native.aoti_load(path, model_name, device_index)
  end

  @doc """
  Run inference on an AOTI model.

  ## Args
    * `model` (`ExTorch.AOTI.Model`) - the loaded model.
    * `inputs` (`[ExTorch.Tensor]`) - list of input tensors.

  ## Returns
  A list of output tensors.
  """
  @spec forward(Model.t(), [ExTorch.Tensor.t()]) :: [ExTorch.Tensor.t()]
  def forward(%Model{} = model, inputs) when is_list(inputs) do
    ExTorch.Native.aoti_forward(model, inputs)
  end

  @doc """
  Get metadata from an AOTI model as a map.

  ## Returns
  A map of `%{String.t() => String.t()}` metadata key-value pairs.
  """
  @spec metadata(Model.t()) :: %{String.t() => String.t()}
  def metadata(%Model{} = model) do
    keys = ExTorch.Native.aoti_get_metadata_keys(model)

    for key <- keys, into: %{} do
      {key, ExTorch.Native.aoti_get_metadata_value(model, key)}
    end
  end

  @doc """
  Get the fully-qualified names of constants (parameters/buffers) in the model.

  ## Returns
  A list of strings like `["fc1.weight", "fc1.bias", ...]`.
  """
  @spec constant_names(Model.t()) :: [String.t()]
  def constant_names(%Model{} = model) do
    ExTorch.Native.aoti_get_constant_fqns(model)
  end

  @libtorch_loaded_key {__MODULE__, :libtorch_loaded}

  # AOTI-compiled .so files inside .pt2 packages link against libtorch.so
  # but don't have rpath set (they're compiled by torch.inductor). Pre-load
  # the libtorch shared libraries with RTLD_GLOBAL so dlopen finds them.
  defp ensure_libtorch_loaded do
    case :persistent_term.get(@libtorch_loaded_key, false) do
      true -> :ok
      false ->
        lib_dir = libtorch_lib_dir()
        for lib <- ["libc10.so", "libtorch_cpu.so", "libtorch.so"] do
          path = Path.join(lib_dir, lib)
          if File.exists?(path) do
            try do
              ExTorch.Native.load_torch_library(path)
            rescue
              _ -> :ok
            end
          end
        end
        # Also load CUDA libs if available
        for lib <- ["libc10_cuda.so", "libtorch_cuda.so"] do
          path = Path.join(lib_dir, lib)
          if File.exists?(path) do
            try do
              ExTorch.Native.load_torch_library(path)
            rescue
              _ -> :ok
            end
          end
        end
        :persistent_term.put(@libtorch_loaded_key, true)
        :ok
    end
  end

  defp libtorch_lib_dir do
    priv = :code.priv_dir(:extorch) |> to_string()
    Path.join([priv, "native", "libtorch", "lib"])
  end
end
