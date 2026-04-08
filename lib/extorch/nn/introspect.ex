defmodule ExTorch.NN.Introspect do
  @moduledoc """
  Introspect the structure of TorchScript (JIT) models.

  This module provides functions to extract model architecture information
  from loaded `.pt` models, including the computation graph, parameter shapes,
  submodule structure, and available methods.

  ## Example

      model = ExTorch.JIT.load("resnet18.pt")
      schema = ExTorch.NN.Introspect.schema(model)
      IO.inspect(schema.submodules, label: "Submodules")

  """

  alias ExTorch.JIT.Model

  defmodule Schema do
    @moduledoc """
    Structured representation of a JIT model's architecture.
    """
    @type param_info :: %{
            name: String.t(),
            shape: [integer()],
            dtype: atom(),
            requires_grad: boolean()
          }

    @type submodule_info :: %{
            name: String.t(),
            type_name: String.t(),
            parameters: [param_info()]
          }

    @type t :: %__MODULE__{
            parameters: [param_info()],
            submodules: [submodule_info()],
            methods: [String.t()]
          }

    defstruct parameters: [],
              submodules: [],
              methods: []
  end

  @doc """
  Extract a structured schema from a JIT model.

  Returns a `%ExTorch.NN.Introspect.Schema{}` with parameter info,
  submodule info, and method names.
  """
  @spec schema(Model.t()) :: Schema.t()
  def schema(%Model{} = model) do
    %Schema{
      parameters: ExTorch.Native.jit_module_parameters_info(model),
      submodules: ExTorch.Native.jit_module_submodules_info(model),
      methods: ExTorch.Native.jit_module_methods_info(model)
    }
  end

  @doc """
  Get the JIT computation graph IR as a string.

  Returns the TorchScript IR for the model's `forward` method,
  showing the operations and data flow.
  """
  @spec graph(Model.t()) :: String.t()
  def graph(%Model{} = model) do
    ExTorch.Native.jit_graph_str(model)
  end

  @doc """
  Generate Elixir source code for an `ExTorch.NN.Module` DSL definition
  that mirrors the structure of a loaded JIT model.

  ## Arguments
    - `model` - A loaded JIT model.
    - `module_name` - The name for the generated Elixir module (default: `"MyModel"`).

  ## Returns
  A string containing Elixir source code.
  """
  @spec to_elixir(Model.t(), String.t()) :: String.t()
  def to_elixir(%Model{} = model, module_name \\ "MyModel") do
    schema = schema(model)
    generate_elixir_source(schema, module_name)
  end

  defp generate_elixir_source(%Schema{} = schema, module_name) do
    layers =
      schema.submodules
      |> Enum.map(fn sub ->
        layer_type = map_type_to_elixir(sub.type_name)
        opts = infer_layer_opts(sub)
        "  deflayer :#{sub.name}, #{layer_type}#{opts}"
      end)
      |> Enum.join("\n")

    """
    defmodule #{module_name} do
      use ExTorch.NN.Module

    #{layers}

      def forward(x) do
    #{generate_forward_chain(schema.submodules)}
      end
    end
    """
  end

  defp map_type_to_elixir(type_name) do
    # Extract the last component of qualified names like "__torch__.torch.nn.modules.linear.Linear"
    short_name =
      type_name
      |> String.split(".")
      |> List.last()

    case short_name do
      "Linear" -> "ExTorch.NN.Linear"
      "Conv1d" -> "ExTorch.NN.Conv1d"
      "Conv2d" -> "ExTorch.NN.Conv2d"
      "BatchNorm1d" -> "ExTorch.NN.BatchNorm1d"
      "BatchNorm2d" -> "ExTorch.NN.BatchNorm2d"
      "LayerNorm" -> "ExTorch.NN.LayerNorm"
      "Dropout" -> "ExTorch.NN.Dropout"
      "ReLU" -> "ExTorch.NN.ReLU"
      "GELU" -> "ExTorch.NN.GELU"
      "Sigmoid" -> "ExTorch.NN.Sigmoid"
      "Tanh" -> "ExTorch.NN.Tanh"
      "Softmax" -> "ExTorch.NN.Softmax"
      "Embedding" -> "ExTorch.NN.Embedding"
      other -> "ExTorch.NN.Unknown(\"#{other}\")"
    end
  end

  defp infer_layer_opts(submodule) do
    params = submodule.parameters

    case {map_type_to_elixir(submodule.type_name), params} do
      {"ExTorch.NN.Linear", [%{name: "weight", shape: [out, inp]} | _]} ->
        ", in_features: #{inp}, out_features: #{out}"

      {"ExTorch.NN.Conv2d", [%{name: "weight", shape: [out, inp, kh, _kw]} | _]} ->
        ", in_channels: #{inp}, out_channels: #{out}, kernel_size: #{kh}"

      {"ExTorch.NN.Conv1d", [%{name: "weight", shape: [out, inp, k]} | _]} ->
        ", in_channels: #{inp}, out_channels: #{out}, kernel_size: #{k}"

      _ ->
        ""
    end
  end

  defp generate_forward_chain(submodules) do
    submodules
    |> Enum.map(fn sub -> "    |> layer(:#{sub.name})" end)
    |> case do
      [] -> "    x"
      [first | rest] -> "    x\n" <> Enum.join([first | rest], "\n")
    end
  end
end
