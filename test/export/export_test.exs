defmodule ExTorchTest.ExportTest do
  use ExUnit.Case, async: true

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    path = Path.join(@fixtures_dir, "simple_mlp_exported.pt2")
    unless File.exists?(path), do: flunk("Run: .venv/bin/python test/fixtures/generate_export_models.py")
    :ok
  end

  describe "read_schema/1" do
    test "reads graph and weights metadata" do
      schema = ExTorch.Export.read_schema(Path.join(@fixtures_dir, "simple_mlp_exported.pt2"))

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
      schema = ExTorch.Export.read_schema(Path.join(@fixtures_dir, "simple_mlp_exported.pt2"))

      assert map_size(schema.weights) == 4
      assert Map.has_key?(schema.weights, "fc1.weight")
      assert schema.weights["fc1.weight"].shape == [20, 10]
      assert schema.weights["fc1.weight"].dtype == :float32
    end
  end

  describe "read_weights/1" do
    test "loads weight tensors" do
      weights = ExTorch.Export.read_weights(Path.join(@fixtures_dir, "simple_mlp_exported.pt2"))

      assert map_size(weights) == 4
      assert %ExTorch.Tensor{} = weights["fc1.weight"]
      assert weights["fc1.weight"].size == {20, 10}
      assert weights["fc1.bias"].size == {20}
      assert weights["fc2.weight"].size == {5, 20}
      assert weights["fc2.bias"].size == {5}
    end
  end

  describe "load/1 and forward/2" do
    test "loads and runs inference on an exported model" do
      model = ExTorch.Export.load(Path.join(@fixtures_dir, "simple_mlp_exported.pt2"))
      assert %ExTorch.Export.Model{} = model
      assert length(model.user_inputs) == 1
      assert length(model.param_inputs) == 4

      input = ExTorch.randn({1, 10})
      output = ExTorch.Export.forward(model, [input])
      assert %ExTorch.Tensor{} = output
      assert output.size == {1, 5}
    end

    test "handles batch inputs" do
      model = ExTorch.Export.load(Path.join(@fixtures_dir, "simple_mlp_exported.pt2"))
      input = ExTorch.randn({8, 10})
      output = ExTorch.Export.forward(model, [input])
      assert output.size == {8, 5}
    end
  end

  describe "to_elixir/2" do
    test "generates valid DSL source" do
      source = ExTorch.Export.to_elixir(
        Path.join(@fixtures_dir, "simple_mlp_exported.pt2"),
        "GeneratedMLP"
      )

      assert String.contains?(source, "defmodule GeneratedMLP do")
      assert String.contains?(source, "use ExTorch.NN.Module")
      assert String.contains?(source, "deflayer :fc1, ExTorch.NN.Linear")
      assert String.contains?(source, "in_features: 10")
      assert String.contains?(source, "out_features: 20")
      assert String.contains?(source, "deflayer :relu, ExTorch.NN.ReLU")
      assert String.contains?(source, "deflayer :fc2, ExTorch.NN.Linear")
    end
  end
end
