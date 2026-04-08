defmodule ExTorch.JIT.ServerTest do
  use ExUnit.Case, async: true

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    mlp_path = Path.join(@fixtures_dir, "simple_mlp.pt")

    unless File.exists?(mlp_path) do
      {_, 0} = System.cmd("python", ["generate_models.py"], cd: @fixtures_dir)
    end

    :ok
  end

  describe "start_link/1" do
    test "starts a model server" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)
      assert Process.alive?(pid)
      GenServer.stop(pid)
    end

    test "starts a named server" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      name = :"test_model_#{:erlang.unique_integer([:positive])}"
      {:ok, _pid} = ExTorch.JIT.Server.start_link(path: path, name: name)
      assert GenServer.whereis(name) != nil
      GenServer.stop(name)
    end
  end

  describe "predict/2" do
    test "runs inference through server" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      input = ExTorch.randn({1, 10})
      output = ExTorch.JIT.Server.predict(pid, [input])

      assert %ExTorch.Tensor{} = output
      assert output.size == {1, 5}

      GenServer.stop(pid)
    end

    test "handles concurrent predictions" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      tasks =
        for _ <- 1..10 do
          Task.async(fn ->
            input = ExTorch.randn({1, 10})
            ExTorch.JIT.Server.predict(pid, [input])
          end)
        end

      results = Task.await_many(tasks, 10_000)

      Enum.each(results, fn output ->
        assert %ExTorch.Tensor{} = output
        assert output.size == {1, 5}
      end)

      GenServer.stop(pid)
    end
  end

  describe "info/1" do
    test "returns model info" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      # Run a prediction first
      input = ExTorch.randn({1, 10})
      ExTorch.JIT.Server.predict(pid, [input])

      info = ExTorch.JIT.Server.info(pid)
      assert info.path == path
      assert info.device == :cpu
      assert info.inference_count == 1
      assert info.uptime_ms >= 0

      GenServer.stop(pid)
    end
  end
end
