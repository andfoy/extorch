defmodule ExTorchTest.MetricsTest do
  use ExUnit.Case

  @fixtures_dir Path.join([__DIR__, "fixtures"])

  setup_all do
    mlp_path = Path.join(@fixtures_dir, "simple_mlp.pt")

    unless File.exists?(mlp_path) do
      {_, 0} = System.cmd("python", ["generate_models.py"], cd: @fixtures_dir)
    end

    :ok
  end

  setup do
    ExTorch.Metrics.setup()
    ExTorch.Metrics.reset_all()
    :ok
  end

  describe "metrics collection" do
    test "records inference count and latency" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      input = ExTorch.randn({1, 10})

      for _ <- 1..5 do
        ExTorch.JIT.Server.predict(pid, [input])
      end

      # Give telemetry handlers a moment to process
      Process.sleep(10)

      metrics = ExTorch.Metrics.get(path)
      assert metrics != nil
      assert metrics.inference_count == 5
      assert metrics.error_count == 0
      assert metrics.total_duration_ms > 0
      assert metrics.min_duration_ms > 0
      assert metrics.max_duration_ms >= metrics.min_duration_ms

      GenServer.stop(pid)
    end

    test "records load duration" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      Process.sleep(10)

      metrics = ExTorch.Metrics.get(path)
      assert metrics != nil
      assert metrics.load_duration_ms > 0

      GenServer.stop(pid)
    end

    test "all/0 returns all tracked models" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      input = ExTorch.randn({1, 10})
      ExTorch.JIT.Server.predict(pid, [input])
      Process.sleep(10)

      all = ExTorch.Metrics.all()
      assert length(all) >= 1
      assert Enum.any?(all, fn {p, _} -> p == path end)

      GenServer.stop(pid)
    end

    test "reset/1 clears metrics for a model" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      input = ExTorch.randn({1, 10})
      ExTorch.JIT.Server.predict(pid, [input])
      Process.sleep(10)

      assert ExTorch.Metrics.get(path) != nil
      ExTorch.Metrics.reset(path)
      assert ExTorch.Metrics.get(path) == nil

      GenServer.stop(pid)
    end
  end

  describe "CUDA monitoring" do
    test "cuda_is_available returns boolean" do
      result = ExTorch.Native.cuda_is_available()
      assert is_boolean(result)
    end

    test "cuda_device_count returns non-negative integer" do
      count = ExTorch.Native.cuda_device_count()
      assert is_integer(count)
      assert count >= 0
    end

    test "memory functions return -1 on CPU-only build" do
      unless ExTorch.Native.cuda_is_available() do
        assert ExTorch.Native.cuda_memory_allocated(0) == -1
        assert ExTorch.Native.cuda_memory_reserved(0) == -1
        assert ExTorch.Native.cuda_max_memory_allocated(0) == -1
      end
    end
  end
end
