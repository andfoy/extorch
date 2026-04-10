defmodule ExTorchTest.ExportTest do
  use ExUnit.Case, async: true

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  @simple_mlp_path Path.join(@fixtures_dir, "simple_mlp_exported.pt2")
  @simple_mlp_input_shape {1, 10}
  @simple_mlp_output_shape {1, 5}

  @convnet_path Path.join(@fixtures_dir, "convnet_exported.pt2")
  @convnet_input_shape {1, 1, 8, 8}
  @convnet_output_shape {1, 3}

  setup_all do
    unless File.exists?(@simple_mlp_path),
      do: flunk("Run: .venv/bin/python test/fixtures/generate_export_models.py")

    # Generated DSL tests recompile a fixed module name across runs.
    Code.put_compiler_option(:ignore_module_conflict, true)
    :ok
  end

  defp load_reference(name, shape) do
    path = Path.join(@fixtures_dir, "#{name}.bin")
    binary = File.read!(path)
    ExTorch.Native.from_binary(binary, shape, :float32)
  end

  describe "read_schema/1" do
    test "reads graph and weights metadata" do
      schema = ExTorch.Export.read_schema(@simple_mlp_path)

      assert is_list(schema.graph)
      assert length(schema.graph) == 3
      assert is_list(schema.inputs)
      assert is_list(schema.outputs)

      # Check graph has correct ops
      targets = Enum.map(schema.graph, & &1.target)
      assert "torch.ops.aten.linear.default" in targets
      assert "torch.ops.aten.relu.default" in targets
    end

    test "reads weight metadata" do
      schema = ExTorch.Export.read_schema(@simple_mlp_path)

      assert map_size(schema.weights) == 4
      assert Map.has_key?(schema.weights, "fc1.weight")
      assert schema.weights["fc1.weight"].shape == [20, 10]
      assert schema.weights["fc1.weight"].dtype == :float32
    end
  end

  describe "read_weights/1" do
    test "loads weight tensors" do
      weights = ExTorch.Export.read_weights(@simple_mlp_path)

      assert map_size(weights) == 4
      assert %ExTorch.Tensor{} = weights["fc1.weight"]
      assert weights["fc1.weight"].size == {20, 10}
      assert weights["fc1.bias"].size == {20}
      assert weights["fc2.weight"].size == {5, 20}
      assert weights["fc2.bias"].size == {5}
    end
  end

  describe "load/1 and forward/2" do
    test "MLP interpreter output matches PyTorch reference" do
      model = ExTorch.Export.load(@simple_mlp_path)
      assert %ExTorch.Export.Model{} = model
      assert length(model.user_inputs) == 1
      assert length(model.param_inputs) == 4

      input = load_reference("simple_mlp_exported_input", @simple_mlp_input_shape)
      expected = load_reference("simple_mlp_exported_output", @simple_mlp_output_shape)

      output = ExTorch.Export.forward(model, [input])
      assert %ExTorch.Tensor{} = output
      assert output.size == @simple_mlp_output_shape

      assert ExTorch.allclose(output, expected, 1.0e-5, 1.0e-6),
             "MLP interpreter output diverged from PyTorch reference"
    end

    test "ConvNet interpreter output matches PyTorch reference" do
      model = ExTorch.Export.load(@convnet_path)

      input = load_reference("convnet_exported_input", @convnet_input_shape)
      expected = load_reference("convnet_exported_output", @convnet_output_shape)

      output = ExTorch.Export.forward(model, [input])
      assert output.size == @convnet_output_shape

      assert ExTorch.allclose(output, expected, 1.0e-5, 1.0e-6),
             "ConvNet interpreter output diverged from PyTorch reference"
    end

    test "handles batch inputs" do
      model = ExTorch.Export.load(@simple_mlp_path)
      input = ExTorch.randn({8, 10})
      output = ExTorch.Export.forward(model, [input])
      assert output.size == {8, 5}
    end
  end

  describe "to_elixir/2" do
    test "generates valid DSL source" do
      source = ExTorch.Export.to_elixir(@simple_mlp_path, "GeneratedMLP")

      assert String.contains?(source, "defmodule GeneratedMLP do")
      assert String.contains?(source, "use ExTorch.NN.Module")
      assert String.contains?(source, "deflayer :fc1, ExTorch.NN.Linear")
      assert String.contains?(source, "in_features: 10")
      assert String.contains?(source, "out_features: 20")
      assert String.contains?(source, "deflayer :relu, ExTorch.NN.ReLU")
      assert String.contains?(source, "deflayer :fc2, ExTorch.NN.Linear")
    end

    test "generated DSL compiles, loads weights, and matches PyTorch reference" do
      source = ExTorch.Export.to_elixir(@simple_mlp_path, "GeneratedMLPNumeric")
      [{module, _bytecode} | _] = Code.compile_string(source)
      assert module == GeneratedMLPNumeric

      model = module.load_weights_from_export(@simple_mlp_path)

      input = load_reference("simple_mlp_exported_input", @simple_mlp_input_shape)
      expected = load_reference("simple_mlp_exported_output", @simple_mlp_output_shape)

      output = module.forward(model, input)
      assert output.size == @simple_mlp_output_shape

      assert ExTorch.allclose(output, expected, 1.0e-5, 1.0e-6),
             "Generated DSL output diverged from PyTorch reference"
    end
  end
end
