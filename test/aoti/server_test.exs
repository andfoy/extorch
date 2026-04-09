defmodule ExTorchTest.AOTI.ServerTest do
  use ExUnit.Case

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    pt2_path = Path.join(@fixtures_dir, "simple_mlp.pt2")
    unless File.exists?(pt2_path), do: flunk("Run generate_aoti_models.py first")
    :ok
  end

  describe "start_link/1" do
    test "starts a server" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      {:ok, pid} = ExTorch.AOTI.Server.start_link(path: path)
      assert Process.alive?(pid)
      GenServer.stop(pid)
    end

    test "starts a named server" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      name = :"aoti_test_#{:erlang.unique_integer([:positive])}"
      {:ok, _} = ExTorch.AOTI.Server.start_link(path: path, name: name)
      assert GenServer.whereis(name) != nil
      GenServer.stop(name)
    end
  end

  describe "predict/2" do
    test "runs inference" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      {:ok, pid} = ExTorch.AOTI.Server.start_link(path: path)

      input = ExTorch.randn({1, 10})
      [output] = ExTorch.AOTI.Server.predict(pid, [input])

      assert %ExTorch.Tensor{} = output
      assert output.size == {1, 5}

      GenServer.stop(pid)
    end

    test "handles concurrent predictions" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      {:ok, pid} = ExTorch.AOTI.Server.start_link(path: path)

      tasks =
        for _ <- 1..10 do
          Task.async(fn ->
            input = ExTorch.randn({1, 10})
            ExTorch.AOTI.Server.predict(pid, [input])
          end)
        end

      results = Task.await_many(tasks, 10_000)

      Enum.each(results, fn [output] ->
        assert %ExTorch.Tensor{} = output
        assert output.size == {1, 5}
      end)

      GenServer.stop(pid)
    end
  end

  describe "info/1" do
    test "returns server info" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      {:ok, pid} = ExTorch.AOTI.Server.start_link(path: path)

      input = ExTorch.randn({1, 10})
      ExTorch.AOTI.Server.predict(pid, [input])

      info = ExTorch.AOTI.Server.info(pid)
      assert info.path == path
      assert info.inference_count == 1
      assert info.error_count == 0
      assert info.uptime_ms >= 0

      GenServer.stop(pid)
    end
  end
end
