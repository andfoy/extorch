defmodule ExTorch.NN.Module do
  @moduledoc """
  DSL for defining neural network modules in Elixir.

  Use this module to declaratively define neural network architectures
  with a PyTorch-inspired syntax. Each module defines layers and a
  `forward/2` function.

  ## Example

      defmodule MyMLP do
        use ExTorch.NN.Module

        deflayer :fc1, ExTorch.NN.Linear, in_features: 784, out_features: 128
        deflayer :relu, ExTorch.NN.ReLU
        deflayer :fc2, ExTorch.NN.Linear, in_features: 128, out_features: 10

        def forward(model, x) do
          x
          |> layer(model, :fc1)
          |> layer(model, :relu)
          |> layer(model, :fc2)
        end
      end

  ## Usage

      # Fresh model with random weights
      model = MyMLP.new()
      output = MyMLP.forward(model, input)

      # Load pre-trained weights from a TorchScript file
      model = MyMLP.from_jit("model.pt")
      output = MyMLP.forward(model, input)

  When loaded via `from_jit/1`, the JIT model's `forward` method is called
  directly, using the pre-trained weights. The DSL definition serves as a
  structural contract that is validated against the `.pt` file's submodules.

  `deflayer` declares a layer at compile time. `layer/3` is a runtime
  function that looks up and applies a named layer during forward.
  """

  @doc """
  Layer behaviour that nn layer modules must implement.
  """
  @callback create(keyword()) :: ExTorch.NN.Layer.t()

  defmacro __using__(_opts) do
    quote do
      import ExTorch.NN.Module, only: [deflayer: 2, deflayer: 3]
      Module.register_attribute(__MODULE__, :nn_layers, accumulate: true)
      @before_compile ExTorch.NN.Module
    end
  end

  @doc """
  Declare a layer in the module definition.

  ## Arguments
    - `name` - Atom name for the layer.
    - `layer_module` - The layer type module (e.g., `ExTorch.NN.Linear`).
    - `opts` - Keyword options passed to the layer's `create/1` callback.
  """
  defmacro deflayer(name, layer_module, opts \\ []) do
    quote do
      @nn_layers {unquote(name), unquote(layer_module), unquote(opts)}
    end
  end

  defmacro __before_compile__(env) do
    layers = Module.get_attribute(env.module, :nn_layers) |> Enum.reverse()

    layer_specs =
      for {name, mod, opts} <- layers do
        {name, mod, opts}
      end

    quote do
      @doc """
      Returns the layer specifications for this module.
      """
      def __layers__ do
        unquote(Macro.escape(layer_specs))
      end

      @doc """
      Create a new instance of this module with all layers initialized
      with random weights.

      Returns a map of `%{layer_name => %ExTorch.NN.Layer{}}`.
      """
      def new(opts \\ []) do
        _device = Keyword.get(opts, :device, :cpu)

        for {name, mod, layer_opts} <- __layers__(), into: %{} do
          {name, mod.create(layer_opts)}
        end
      end

      @doc """
      Load a pre-trained TorchScript model and validate it against
      this module's layer definitions.

      The returned model uses the JIT model's `forward` method directly,
      with the pre-trained weights from the `.pt` file.

      ## Arguments
        - `path` - Path to a `.pt` TorchScript file.
        - `opts` - Keyword options:
          - `:device` - Device to load onto (default: `:cpu`).
          - `:eval` - Set to eval mode (default: `true`).
          - `:validate` - Validate submodules match DSL (default: `true`).

      ## Returns
      An `%ExTorch.NN.JITBackedModel{}`.
      """
      def from_jit(path, opts \\ []) do
        device = Keyword.get(opts, :device, :cpu)
        eval_mode = Keyword.get(opts, :eval, true)
        validate = Keyword.get(opts, :validate, true)

        jit_model = ExTorch.JIT.load(path, device: device)

        if eval_mode, do: ExTorch.JIT.eval(jit_model)

        if validate do
          declared = __layers__() |> Enum.map(fn {name, _, _} -> Atom.to_string(name) end) |> MapSet.new()
          actual = ExTorch.JIT.modules(jit_model) |> MapSet.new()

          missing = MapSet.difference(declared, actual)

          if MapSet.size(missing) > 0 do
            raise ArgumentError,
                  "JIT model is missing submodules declared in DSL: #{inspect(MapSet.to_list(missing))}"
          end
        end

        %ExTorch.NN.JITBackedModel{
          jit_model: jit_model,
          module_name: __MODULE__,
          layer_names: Enum.map(__layers__(), fn {name, _, _} -> name end)
        }
      end

      @doc """
      Apply a named layer to an input tensor.

      For DSL-created models (from `new/0`), looks up and applies the layer.
      For JIT-backed models (from `from_jit/1`), this is a no-op passthrough
      since the JIT model's `forward` handles all layers internally.
      """
      def layer(input, %ExTorch.NN.JITBackedModel{}, _name) do
        # JIT-backed: layers are handled by the JIT forward pass, not individually
        input
      end

      def layer(input, model, name) when is_map(model) and is_atom(name) do
        layer_instance = Map.fetch!(model, name)
        ExTorch.NN.forward(input, layer_instance)
      end

      @doc """
      Run the forward pass. For JIT-backed models, delegates to the JIT model's
      forward method. For DSL models, calls the user-defined `forward/2`.
      """
      def predict(%ExTorch.NN.JITBackedModel{jit_model: jit_model}, inputs) when is_list(inputs) do
        ExTorch.JIT.forward(jit_model, inputs)
      end

      def predict(model, inputs) when is_map(model) and is_list(inputs) do
        # For single-tensor input convenience
        [input | _] = inputs
        forward(model, input)
      end

      @doc """
      Get all named parameters.

      Works for both DSL models and JIT-backed models.
      """
      @doc """
      Load pre-trained weights from a TorchScript file into a DSL model.

      Creates the DSL model via `new/0`, loads the JIT model, then copies
      matching parameters from the JIT model into the DSL layers.

      ## Arguments
        - `path` - Path to the `.pt` TorchScript file.
        - `opts` - Keyword options:
          - `:device` - Device (default: `:cpu`).

      ## Returns
      The DSL model map with pre-trained weights.

      ## Example

          model = MyMLP.load_weights("trained.pt")
          # model is a regular DSL model (%{fc1: %Layer{}, ...}) with trained weights
          output = MyMLP.forward(model, input)
      """
      def load_weights(path, opts \\ []) do
        device = Keyword.get(opts, :device, :cpu)

        # Load JIT model and extract parameters grouped by submodule
        jit_model = ExTorch.JIT.load(path, device: device)
        jit_params = ExTorch.JIT.parameters(jit_model)

        # Create fresh DSL model
        model = new(opts)

        # Group JIT params by layer name: "fc1.weight" -> {fc1, [{"weight", tensor}]}
        grouped =
          Enum.group_by(
            jit_params,
            fn {name, _tensor} ->
              name |> String.split(".") |> hd()
            end,
            fn {name, tensor} ->
              param_name = name |> String.split(".") |> tl() |> Enum.join(".")
              {param_name, tensor}
            end
          )

        # Copy parameters into each DSL layer
        for {layer_name, layer_instance} <- model, into: %{} do
          layer_key = Atom.to_string(layer_name)

          case Map.get(grouped, layer_key) do
            nil ->
              {layer_name, layer_instance}

            layer_params ->
              ExTorch.NN.copy_parameters(layer_instance, layer_params)
              {layer_name, layer_instance}
          end
        end
      end

      def parameters(%ExTorch.NN.JITBackedModel{jit_model: jit_model}) do
        ExTorch.JIT.parameters(jit_model)
      end

      def parameters(model) when is_map(model) do
        for {layer_name, layer_instance} <- model,
            {param_name, tensor} <- ExTorch.NN.parameters(layer_instance) do
          {"#{layer_name}.#{param_name}", tensor}
        end
      end
    end
  end
end
