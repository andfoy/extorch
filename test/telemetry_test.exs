defmodule ExTorchTest.TelemetryTest do
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
    # Attach telemetry handlers for this test
    ref = make_ref()
    pid = self()

    handler_id = "test-#{inspect(ref)}"

    :telemetry.attach_many(
      handler_id,
      [
        [:extorch, :jit, :load, :start],
        [:extorch, :jit, :load, :stop],
        [:extorch, :jit, :forward, :start],
        [:extorch, :jit, :forward, :stop]
      ],
      fn event, measurements, metadata, _ ->
        send(pid, {:telemetry, event, measurements, metadata})
      end,
      nil
    )

    on_exit(fn -> :telemetry.detach(handler_id) end)
    :ok
  end

  describe "JIT.Server telemetry" do
    test "emits load start/stop events" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      assert_receive {:telemetry, [:extorch, :jit, :load, :start], %{system_time: _}, %{path: ^path}}
      assert_receive {:telemetry, [:extorch, :jit, :load, :stop], %{duration: duration}, %{path: ^path}}
      assert is_integer(duration)
      assert duration > 0

      GenServer.stop(pid)
    end

    test "emits forward start/stop events on predict" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt")
      {:ok, pid} = ExTorch.JIT.Server.start_link(path: path)

      # Drain load events
      assert_receive {:telemetry, [:extorch, :jit, :load, :start], _, _}
      assert_receive {:telemetry, [:extorch, :jit, :load, :stop], _, _}

      input = ExTorch.randn({1, 10})
      _output = ExTorch.JIT.Server.predict(pid, [input])

      assert_receive {:telemetry, [:extorch, :jit, :forward, :start], %{system_time: _}, metadata}
      assert metadata.path == path
      assert metadata.input_count == 1

      assert_receive {:telemetry, [:extorch, :jit, :forward, :stop], %{duration: duration}, _}
      assert is_integer(duration)
      assert duration > 0

      GenServer.stop(pid)
    end
  end
end
