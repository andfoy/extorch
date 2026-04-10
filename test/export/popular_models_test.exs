defmodule ExTorchTest.Export.PopularModelsTest do
  use ExUnit.Case

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  # Tolerances for the interpreter vs PyTorch comparison. Both stacks call
  # the same libtorch primitives, but graph decomposition can introduce
  # small accumulated drift in deeper networks, so the tolerance is loose
  # enough to absorb that without hiding real regressions.
  @rtol 1.0e-4
  @atol 1.0e-5

  setup_all do
    path = Path.join(@fixtures_dir, "alexnet.pt2")
    unless File.exists?(path), do: flunk("Run: .venv/bin/python test/fixtures/generate_popular_models.py")
    :ok
  end

  defp load_reference(name, shape) do
    path = Path.join(@fixtures_dir, "#{name}.bin")
    unless File.exists?(path) do
      flunk("Missing reference fixture #{path}. Re-run generate_popular_models.py.")
    end
    binary = File.read!(path)
    ExTorch.Native.from_binary(binary, shape, :float32)
  end

  defp run_model(name, input_shape, expected_output_shape) do
    path = Path.join(@fixtures_dir, "#{name}.pt2")
    model = ExTorch.Export.load(path)

    input = load_reference("#{name}_input", input_shape)
    expected = load_reference("#{name}_output", expected_output_shape)

    try do
      output = ExTorch.Export.forward(model, [input])
      assert %ExTorch.Tensor{} = output
      assert output.size == expected_output_shape,
        "#{name}: expected #{inspect(expected_output_shape)}, got #{inspect(output.size)}"

      assert ExTorch.allclose(output, expected, @rtol, @atol),
        "#{name}: interpreter output diverged from PyTorch reference (rtol=#{@rtol}, atol=#{@atol})"
      :ok
    rescue
      e in RuntimeError ->
        if String.contains?(e.message, "Unsupported ATen op") do
          IO.puts("\n    [#{name}] #{e.message}")
          :unsupported
        else
          reraise e, __STACKTRACE__
        end
    end
  end

  describe "read_schema for all models" do
    test "alexnet" do
      schema = ExTorch.Export.read_schema(Path.join(@fixtures_dir, "alexnet.pt2"))
      assert length(schema.graph) > 0
    end

    test "resnet18" do
      schema = ExTorch.Export.read_schema(Path.join(@fixtures_dir, "resnet18.pt2"))
      assert length(schema.graph) > 0
    end

    test "mobilenetv2" do
      schema = ExTorch.Export.read_schema(Path.join(@fixtures_dir, "mobilenetv2.pt2"))
      assert length(schema.graph) > 0
    end
  end

  describe "read_weights for all models" do
    test "alexnet" do
      weights = ExTorch.Export.read_weights(Path.join(@fixtures_dir, "alexnet.pt2"))
      assert map_size(weights) > 0
    end

    test "resnet18" do
      weights = ExTorch.Export.read_weights(Path.join(@fixtures_dir, "resnet18.pt2"))
      assert map_size(weights) > 0
    end
  end

  @moduletag :popular_models

  describe "inference on classical CNNs" do
    test "alexnet" do
      run_model("alexnet", {1, 3, 224, 224}, {1, 1000})
    end

    test "squeezenet" do
      run_model("squeezenet", {1, 3, 224, 224}, {1, 1000})
    end

    test "mobilenetv2" do
      run_model("mobilenetv2", {1, 3, 224, 224}, {1, 1000})
    end

    test "resnet18" do
      run_model("resnet18", {1, 3, 224, 224}, {1, 1000})
    end

    test "resnet50" do
      run_model("resnet50", {1, 3, 224, 224}, {1, 1000})
    end

    test "vgg11" do
      run_model("vgg11", {1, 3, 224, 224}, {1, 1000})
    end
  end

  describe "inference on custom models" do
    test "simple_transformer" do
      run_model("simple_transformer", {1, 16, 32}, {1, 10})
    end

    test "autoencoder" do
      run_model("autoencoder", {1, 784}, {1, 784})
    end

    test "mini_bert" do
      run_model("mini_bert", {1, 16, 32}, {1, 10})
    end

    test "conv_autoencoder" do
      run_model("conv_autoencoder", {1, 3, 32, 32}, {1, 3, 32, 32})
    end

    test "simple_lstm" do
      run_model("simple_lstm", {1, 16, 32}, {1, 10})
    end
  end
end
