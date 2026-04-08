defmodule ExTorch.JIT.LoadTest do
  use ExUnit.Case, async: true

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    # Generate test models if they don't exist
    mlp_path = Path.join(@fixtures_dir, "simple_mlp.pt")

    unless File.exists?(mlp_path) do
      {_, 0} = System.cmd("python", ["generate_models.py"], cd: @fixtures_dir)
    end

    :ok
  end

  describe "load/2" do
    test "loads a TorchScript model from file" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      assert %ExTorch.JIT.Model{} = model
      assert model.device == :cpu
    end

    test "loads a model with explicit device" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path, device: :cpu)
      assert %ExTorch.JIT.Model{} = model
      assert model.device == :cpu
    end

    test "raises on invalid path" do
      assert_raise ErlangError, fn ->
        ExTorch.JIT.load("/nonexistent/path/model.pt")
      end
    end
  end

  describe "save/2" do
    test "saves and reloads a model" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)

      tmp_path = Path.join(System.tmp_dir!(), "extorch_test_save_#{:erlang.unique_integer([:positive])}.pt")

      try do
        ExTorch.JIT.save(model, tmp_path)
        assert File.exists?(tmp_path)

        reloaded = ExTorch.JIT.load(tmp_path)
        assert %ExTorch.JIT.Model{} = reloaded
      after
        File.rm(tmp_path)
      end
    end
  end

  describe "methods/1" do
    test "lists model methods" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      methods = ExTorch.JIT.methods(model)
      assert is_list(methods)
      assert "forward" in methods
    end
  end

  describe "parameters/1" do
    test "returns named parameters" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      params = ExTorch.JIT.parameters(model)
      assert is_list(params)
      assert length(params) > 0

      Enum.each(params, fn {name, tensor} ->
        assert is_binary(name)
        assert %ExTorch.Tensor{} = tensor
      end)
    end
  end

  describe "buffers/1" do
    test "returns named buffers (empty for MLP)" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      buffers = ExTorch.JIT.buffers(model)
      assert is_list(buffers)
    end
  end

  describe "modules/1" do
    test "returns submodule names" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      modules = ExTorch.JIT.modules(model)
      assert is_list(modules)
      assert length(modules) > 0
    end
  end

  describe "eval/1 and train/1" do
    test "sets model mode without error" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      assert :ok = ExTorch.JIT.eval(model)
      assert :ok = ExTorch.JIT.train(model)
    end
  end

  describe "to/2" do
    test "moves model to cpu" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      model = ExTorch.JIT.load(path)
      moved = ExTorch.JIT.to(model, :cpu)
      assert %ExTorch.JIT.Model{} = moved
      assert moved.device == :cpu
    end
  end
end
