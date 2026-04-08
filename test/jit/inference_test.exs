defmodule ExTorch.JIT.InferenceTest do
  use ExUnit.Case, async: true

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    mlp_path = Path.join(@fixtures_dir, "simple_mlp.pt")

    unless File.exists?(mlp_path) do
      {_, 0} = System.cmd("python", ["generate_models.py"], cd: @fixtures_dir)
    end

    :ok
  end

  describe "forward/2 with tensor output" do
    test "runs inference on simple MLP" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      ExTorch.JIT.eval(model)

      input = ExTorch.randn({1, 10})
      output = ExTorch.JIT.forward(model, [input])

      assert %ExTorch.Tensor{} = output
      assert output.size == {1, 5}
    end

    test "handles batch inference" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      ExTorch.JIT.eval(model)

      input = ExTorch.randn({4, 10})
      output = ExTorch.JIT.forward(model, [input])

      assert %ExTorch.Tensor{} = output
      assert output.size == {4, 5}
    end
  end

  describe "forward/2 with tuple output" do
    test "returns Elixir tuple of tensors" do
      path = Path.join(@fixtures_dir, "tuple_model.pt")
      model = ExTorch.JIT.load(path)
      ExTorch.JIT.eval(model)

      input = ExTorch.randn({1, 10})
      output = ExTorch.JIT.forward(model, [input])

      assert is_tuple(output)
      assert tuple_size(output) == 2

      {t1, t2} = output
      assert %ExTorch.Tensor{} = t1
      assert %ExTorch.Tensor{} = t2
      assert t1.size == {1, 5}
      assert t2.size == {1, 3}
    end
  end

  describe "forward/2 with dict output" do
    test "returns Elixir map of tensors" do
      path = Path.join(@fixtures_dir, "dict_model.pt")
      model = ExTorch.JIT.load(path)
      ExTorch.JIT.eval(model)

      input = ExTorch.randn({1, 10})
      output = ExTorch.JIT.forward(model, [input])

      assert is_map(output)
      assert Map.has_key?(output, "logits")
      assert Map.has_key?(output, "features")

      assert %ExTorch.Tensor{} = output["logits"]
      assert %ExTorch.Tensor{} = output["features"]
      assert output["logits"].size == {1, 5}
      assert output["features"].size == {1, 3}
    end
  end

  describe "forward/2 with multiple inputs" do
    test "passes multiple tensors to model" do
      path = Path.join(@fixtures_dir, "multi_input_model.pt")
      model = ExTorch.JIT.load(path)
      ExTorch.JIT.eval(model)

      x = ExTorch.randn({1, 10})
      y = ExTorch.randn({1, 8})
      output = ExTorch.JIT.forward(model, [x, y])

      assert %ExTorch.Tensor{} = output
      assert output.size == {1, 5}
    end
  end

  describe "invoke/3" do
    test "invokes the forward method by name" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      ExTorch.JIT.eval(model)

      input = ExTorch.randn({1, 10})
      output = ExTorch.JIT.invoke(model, "forward", [input])

      assert %ExTorch.Tensor{} = output
      assert output.size == {1, 5}
    end
  end
end
