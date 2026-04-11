# Telemetry and Metrics — Observability for Model Serving
#
# Demonstrates ExTorch's built-in telemetry events and the Metrics
# module for tracking inference latency, throughput, and errors.
#
# Usage: mix run examples/serving/telemetry_dashboard.exs

defmodule TelemetryDemo do
  @fixtures Path.join([__DIR__, "..", "..", "test", "fixtures"])

  def run do
    ExTorch.set_grad_enabled(false)

    path = Path.join(@fixtures, "resnet18.pt2")
    bin = Path.join(@fixtures, "resnet18_input.bin")

    unless File.exists?(path) do
      IO.puts("Missing fixtures. Run: .venv/bin/python test/fixtures/generate_popular_models.py")
      System.halt(1)
    end

    device = if ExTorch.Native.cuda_is_available(), do: :cuda, else: :cpu

    # =========================================================================
    # 1. Attach custom telemetry handlers
    # =========================================================================
    IO.puts("=== Custom Telemetry Handlers ===\n")

    :telemetry.attach_many("demo-logger", [
      [:extorch, :export, :load, :start],
      [:extorch, :export, :load, :stop],
      [:extorch, :export, :forward, :start],
      [:extorch, :export, :forward, :stop]
    ], fn event, measurements, metadata, _config ->
      event_name = Enum.join(event, ".")
      case measurements do
        %{duration: duration} ->
          ms = Float.round(duration / 1_000_000, 2)
          IO.puts("  [telemetry] #{event_name}: #{ms}ms (#{inspect(Map.keys(metadata))})")
        _ ->
          IO.puts("  [telemetry] #{event_name}: started")
      end
    end, nil)

    # =========================================================================
    # 2. Use the built-in ExTorch.Export.Server with telemetry
    # =========================================================================
    IO.puts("Starting Export server...")
    {:ok, server} = ExTorch.Export.Server.start_link(path: path, device: device, name: :resnet)

    input = ExTorch.Native.from_binary(File.read!(bin), {1, 3, 224, 224}, :float32)
    input = if device == :cuda, do: ExTorch.Tensor.to(input, device: :cuda), else: input

    IO.puts("\nRunning 20 inferences...\n")
    latencies =
      for _ <- 1..20 do
        {us, {:ok, _output}} = :timer.tc(fn ->
          ExTorch.Export.Server.predict(:resnet, [input])
        end)
        us / 1000.0
      end

    # =========================================================================
    # 3. Report statistics
    # =========================================================================
    sorted = Enum.sort(latencies)
    IO.puts("\n=== Inference Statistics ===")
    IO.puts("  Requests:  #{length(latencies)}")
    IO.puts("  p50:       #{Float.round(Enum.at(sorted, 9), 2)}ms")
    IO.puts("  p95:       #{Float.round(Enum.at(sorted, 18), 2)}ms")
    IO.puts("  min:       #{Float.round(List.first(sorted), 2)}ms")
    IO.puts("  max:       #{Float.round(List.last(sorted), 2)}ms")
    IO.puts("  mean:      #{Float.round(Enum.sum(latencies) / length(latencies), 2)}ms")

    throughput = length(latencies) / (Enum.sum(latencies) / 1000.0)
    IO.puts("  throughput: #{Float.round(throughput, 1)} inferences/sec")

    # =========================================================================
    # 4. Server info
    # =========================================================================
    info = ExTorch.Export.Server.info(:resnet)
    IO.puts("\n=== Server Info ===")
    IO.puts("  #{inspect(info)}")

    :telemetry.detach("demo-logger")
  end
end

TelemetryDemo.run()
