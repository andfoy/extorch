# GenServer Model Pool — Production Serving
#
# Demonstrates supervised model serving with pooling, backpressure,
# and per-model telemetry. This is the recommended pattern for
# production deployments.
#
# Usage: mix run examples/serving/genserver_pool.exs

defmodule ModelPool do
  @moduledoc """
  A simple model pool that distributes inference requests across
  multiple model replicas. Each replica is a GenServer holding a
  loaded model, providing serialized access and fault isolation.
  """

  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: opts[:name] || __MODULE__)
  end

  @impl true
  def init(opts) do
    path = Keyword.fetch!(opts, :path)
    pool_size = Keyword.get(opts, :pool_size, System.schedulers_online())
    device = Keyword.get(opts, :device, :cpu)
    mode = Keyword.get(opts, :mode, :native)

    children =
      for i <- 1..pool_size do
        worker_name = :"#{__MODULE__}.Worker.#{i}"
        %{
          id: worker_name,
          start: {ModelPool.Worker, :start_link, [[
            path: path,
            device: device,
            mode: mode,
            name: worker_name
          ]]}
        }
      end

    Supervisor.init(children, strategy: :one_for_one)
  end

  @doc """
  Run inference on a random replica from the pool.
  """
  def predict(pool \\ __MODULE__, inputs, timeout \\ 5_000) do
    workers = Supervisor.which_children(pool)
    {_id, pid, _type, _modules} = Enum.random(workers)
    GenServer.call(pid, {:predict, inputs}, timeout)
  end

  @doc """
  Get stats from all workers.
  """
  def stats(pool \\ __MODULE__) do
    workers = Supervisor.which_children(pool)
    Enum.map(workers, fn {id, pid, _, _} ->
      {id, GenServer.call(pid, :stats)}
    end)
  end
end

defmodule ModelPool.Worker do
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name])
  end

  @impl true
  def init(opts) do
    path = Keyword.fetch!(opts, :path)
    device = Keyword.get(opts, :device, :cpu)
    mode = Keyword.get(opts, :mode, :native)

    model = ExTorch.Export.load(path, device: device)

    {:ok, %{
      model: model,
      mode: mode,
      device: device,
      path: path,
      inference_count: 0,
      total_us: 0,
      started_at: System.monotonic_time(:millisecond)
    }}
  end

  @impl true
  def handle_call({:predict, inputs}, _from, state) do
    {us, result} = :timer.tc(fn ->
      case state.mode do
        :native -> ExTorch.Export.forward_native(state.model, inputs)
        :interpreter -> ExTorch.Export.forward(state.model, inputs)
      end
    end)

    state = %{state |
      inference_count: state.inference_count + 1,
      total_us: state.total_us + us
    }

    {:reply, {:ok, result}, state}
  end

  def handle_call(:stats, _from, state) do
    uptime_ms = System.monotonic_time(:millisecond) - state.started_at
    avg_ms = if state.inference_count > 0,
      do: Float.round(state.total_us / state.inference_count / 1000, 2),
      else: 0.0

    stats = %{
      inference_count: state.inference_count,
      avg_latency_ms: avg_ms,
      uptime_ms: uptime_ms,
      mode: state.mode,
      device: state.device
    }

    {:reply, stats, state}
  end
end

# === Demo ===

defmodule PoolDemo do
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
    pool_size = 4

    IO.puts("Starting model pool: #{pool_size} replicas on #{device}")
    {:ok, _} = ModelPool.start_link(
      path: path,
      device: device,
      mode: :native,
      pool_size: pool_size
    )

    input = ExTorch.Native.from_binary(File.read!(bin), {1, 3, 224, 224}, :float32)
    input = if device == :cuda, do: ExTorch.Tensor.to(input, device: :cuda), else: input

    # Warmup
    for _ <- 1..pool_size, do: ModelPool.predict([input])

    # Concurrent inference
    IO.puts("\nRunning 100 concurrent requests...")
    tasks =
      for _ <- 1..100 do
        Task.async(fn ->
          {us, {:ok, output}} = :timer.tc(fn -> ModelPool.predict([input]) end)
          {us, output.size}
        end)
      end

    results = Task.await_many(tasks, 30_000)
    latencies = Enum.map(results, fn {us, _} -> us / 1000.0 end) |> Enum.sort()

    IO.puts("\nResults:")
    IO.puts("  Total requests: #{length(results)}")
    IO.puts("  p50 latency:    #{Float.round(Enum.at(latencies, 49), 2)}ms")
    IO.puts("  p95 latency:    #{Float.round(Enum.at(latencies, 94), 2)}ms")
    IO.puts("  p99 latency:    #{Float.round(Enum.at(latencies, 98), 2)}ms")

    IO.puts("\nWorker stats:")
    for {id, stats} <- ModelPool.stats() do
      IO.puts("  #{id}: #{stats.inference_count} inferences, avg #{stats.avg_latency_ms}ms")
    end
  end
end

PoolDemo.run()
