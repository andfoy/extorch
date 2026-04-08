defmodule ExTorchTest.NN.ModuleDSLTest do
  use ExUnit.Case, async: true

  defmodule SimpleMLP do
    use ExTorch.NN.Module

    deflayer :fc1, ExTorch.NN.Linear, in_features: 10, out_features: 20
    deflayer :relu, ExTorch.NN.ReLU
    deflayer :fc2, ExTorch.NN.Linear, in_features: 20, out_features: 5

    def forward(model, x) do
      x
      |> layer(model, :fc1)
      |> layer(model, :relu)
      |> layer(model, :fc2)
    end
  end

  defmodule ConvNet do
    use ExTorch.NN.Module

    deflayer :conv1, ExTorch.NN.Conv2d, in_channels: 1, out_channels: 8, kernel_size: 3
    deflayer :relu, ExTorch.NN.ReLU

    def forward(model, x) do
      x
      |> layer(model, :conv1)
      |> layer(model, :relu)
    end
  end

  describe "__layers__/0" do
    test "returns layer specifications" do
      layers = SimpleMLP.__layers__()
      assert length(layers) == 3

      [{:fc1, ExTorch.NN.Linear, fc1_opts}, {:relu, ExTorch.NN.ReLU, []}, {:fc2, ExTorch.NN.Linear, fc2_opts}] = layers
      assert fc1_opts[:in_features] == 10
      assert fc1_opts[:out_features] == 20
      assert fc2_opts[:in_features] == 20
      assert fc2_opts[:out_features] == 5
    end
  end

  describe "new/0" do
    test "creates all layers" do
      model = SimpleMLP.new()
      assert is_map(model)
      assert Map.has_key?(model, :fc1)
      assert Map.has_key?(model, :relu)
      assert Map.has_key?(model, :fc2)

      assert %ExTorch.NN.Layer{type_name: "Linear"} = model[:fc1]
      assert %ExTorch.NN.Layer{type_name: "ReLU"} = model[:relu]
      assert %ExTorch.NN.Layer{type_name: "Linear"} = model[:fc2]
    end
  end

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  describe "from_jit/1" do
    test "loads a JIT model and validates against DSL" do
      model = SimpleMLP.from_jit(Path.join(@fixtures_dir, "simple_mlp.pt"))
      assert %ExTorch.NN.JITBackedModel{} = model
      assert model.module_name == SimpleMLP
      assert model.layer_names == [:fc1, :relu, :fc2]
    end

    test "predict works on JIT-backed model" do
      model = SimpleMLP.from_jit(Path.join(@fixtures_dir, "simple_mlp.pt"))
      input = ExTorch.randn({1, 10})
      output = SimpleMLP.predict(model, [input])
      assert output.size == {1, 5}
    end

    test "parameters works on JIT-backed model" do
      model = SimpleMLP.from_jit(Path.join(@fixtures_dir, "simple_mlp.pt"))
      params = SimpleMLP.parameters(model)
      assert length(params) == 4
    end

    test "raises on architecture mismatch" do
      assert_raise ArgumentError, ~r/missing submodules/, fn ->
        ConvNet.from_jit(Path.join(@fixtures_dir, "simple_mlp.pt"))
      end
    end
  end

  describe "load_weights/1" do
    test "loads weights and produces same output as JIT" do
      input = ExTorch.randn({1, 10})
      path = Path.join(@fixtures_dir, "simple_mlp.pt")

      loaded = SimpleMLP.load_weights(path)
      out_loaded = SimpleMLP.forward(loaded, input)

      jit = ExTorch.JIT.load(path)
      ExTorch.JIT.eval(jit)
      out_jit = ExTorch.JIT.forward(jit, [input])

      assert ExTorch.allclose(out_loaded, out_jit)
    end

    test "returns a regular DSL model map" do
      loaded = SimpleMLP.load_weights(Path.join(@fixtures_dir, "simple_mlp.pt"))
      assert is_map(loaded)
      assert %ExTorch.NN.Layer{} = loaded[:fc1]
      assert %ExTorch.NN.Layer{} = loaded[:fc2]
    end
  end

  describe "forward/2" do
    test "MLP forward produces correct output shape" do
      model = SimpleMLP.new()
      input = ExTorch.randn({4, 10})
      output = SimpleMLP.forward(model, input)

      assert %ExTorch.Tensor{} = output
      assert output.size == {4, 5}
    end

    test "ConvNet forward produces correct output shape" do
      model = ConvNet.new()
      input = ExTorch.randn({1, 1, 8, 8})
      output = ConvNet.forward(model, input)

      assert %ExTorch.Tensor{} = output
      # Conv2d with kernel=3, no padding: 8-3+1=6
      assert output.size == {1, 8, 6, 6}
    end
  end
end
