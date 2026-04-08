defmodule ExTorchTest.NN.IntrospectTest do
  use ExUnit.Case, async: true

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    mlp_path = Path.join(@fixtures_dir, "simple_mlp.pt")

    unless File.exists?(mlp_path) do
      {_, 0} = System.cmd("python", ["generate_models.py"], cd: @fixtures_dir)
    end

    :ok
  end

  describe "schema/1" do
    test "extracts parameters from a model" do
      model = ExTorch.JIT.load(Path.join(@fixtures_dir, "simple_mlp.pt"))
      schema = ExTorch.NN.Introspect.schema(model)

      assert length(schema.parameters) > 0

      Enum.each(schema.parameters, fn p ->
        assert is_binary(p.name)
        assert is_list(p.shape)
        assert is_atom(p.dtype)
        assert is_boolean(p.requires_grad)
      end)
    end

    test "extracts submodules from a model" do
      model = ExTorch.JIT.load(Path.join(@fixtures_dir, "simple_mlp.pt"))
      schema = ExTorch.NN.Introspect.schema(model)

      assert length(schema.submodules) == 3

      names = Enum.map(schema.submodules, & &1.name)
      assert "fc1" in names
      assert "relu" in names
      assert "fc2" in names
    end

    test "extracts methods from a model" do
      model = ExTorch.JIT.load(Path.join(@fixtures_dir, "simple_mlp.pt"))
      schema = ExTorch.NN.Introspect.schema(model)

      assert "forward" in schema.methods
    end

    test "submodule parameters have correct shapes" do
      model = ExTorch.JIT.load(Path.join(@fixtures_dir, "simple_mlp.pt"))
      schema = ExTorch.NN.Introspect.schema(model)

      fc1 = Enum.find(schema.submodules, &(&1.name == "fc1"))
      weight = Enum.find(fc1.parameters, &(&1.name == "weight"))
      assert weight.shape == [20, 10]
    end
  end

  describe "graph/1" do
    test "returns JIT graph IR string" do
      model = ExTorch.JIT.load(Path.join(@fixtures_dir, "simple_mlp.pt"))
      graph = ExTorch.NN.Introspect.graph(model)

      assert is_binary(graph)
      assert String.contains?(graph, "graph(")
      assert String.contains?(graph, "Tensor")
    end
  end

  describe "to_elixir/2" do
    test "generates valid-looking Elixir source code" do
      model = ExTorch.JIT.load(Path.join(@fixtures_dir, "simple_mlp.pt"))
      source = ExTorch.NN.Introspect.to_elixir(model, "GeneratedMLP")

      assert String.contains?(source, "defmodule GeneratedMLP do")
      assert String.contains?(source, "use ExTorch.NN.Module")
      assert String.contains?(source, "deflayer :fc1")
      assert String.contains?(source, "deflayer :relu")
      assert String.contains?(source, "deflayer :fc2")
      assert String.contains?(source, "ExTorch.NN.Linear")
      assert String.contains?(source, "in_features: 10")
      assert String.contains?(source, "out_features: 20")
    end
  end
end
