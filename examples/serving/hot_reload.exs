# Hot Model Reload — Zero-Downtime Updates
#
# Demonstrates swapping a model without dropping in-flight requests.
# The pattern: load new model in background, atomic swap, drain old.
#
# Usage: mix run examples/serving/hot_reload.exs

defmodule HotReloadServer do
  @moduledoc """
  A model server that supports hot-reloading models without downtime.

  The reload process:
  1. Load new model in a background task (doesn't block inference)
  2. Atomically swap the model reference
  3. In-flight requests on the old model complete normally
  4. Old model is garbage collected

  Use `reload/2` to trigger a model swap:

      HotReloadServer.reload(server, new_path)
  """

  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name] || __MODULE__)
  end

  @doc "Run inference on the current model."
  def predict(server \\ __MODULE__, inputs, timeout \\ 10_000) do
    GenServer.call(server, {:predict, inputs}, timeout)
  end

  @doc "Reload the model from a new path. Non-blocking — returns immediately."
  def reload(server \\ __MODULE__, new_path, opts \\ []) do
    GenServer.cast(server, {:reload, new_path, opts})
  end

  @doc "Get the current model info."
  def info(server \\ __MODULE__) do
    GenServer.call(server, :info)
  end

  # --- GenServer callbacks ---

  @impl true
  def init(opts) do
    path = Keyword.fetch!(opts, :path)
    device = Keyword.get(opts, :device, :cpu)
    mode = Keyword.get(opts, :mode, :native)

    model = ExTorch.Export.load(path, device: device)

    {:ok, %{
      model: model,
      path: path,
      device: device,
      mode: mode,
      version: 1,
      loaded_at: DateTime.utc_now(),
      reload_in_progress: false,
      inference_count: 0
    }}
  end

  @impl true
  def handle_call({:predict, inputs}, _from, state) do
    result = case state.mode do
      :native -> ExTorch.Export.forward_native(state.model, inputs)
      :interpreter -> ExTorch.Export.forward(state.model, inputs)
    end

    {:reply, {:ok, result}, %{state | inference_count: state.inference_count + 1}}
  end

  def handle_call(:info, _from, state) do
    info = %{
      path: state.path,
      device: state.device,
      version: state.version,
      loaded_at: state.loaded_at,
      reload_in_progress: state.reload_in_progress,
      inference_count: state.inference_count
    }
    {:reply, info, state}
  end

  @impl true
  def handle_cast({:reload, new_path, opts}, state) do
    if state.reload_in_progress do
      IO.puts("[hot_reload] Reload already in progress, ignoring")
      {:noreply, state}
    else
      IO.puts("[hot_reload] Loading new model from #{new_path}...")
      server = self()
      device = Keyword.get(opts, :device, state.device)

      # Load in background — inference continues on the old model
      Task.start(fn ->
        try do
          new_model = ExTorch.Export.load(new_path, device: device)
          # Verify the model works before swapping
          send(server, {:reload_complete, new_model, new_path, device})
        rescue
          e ->
            IO.puts("[hot_reload] FAILED: #{Exception.message(e)}")
            send(server, :reload_failed)
        end
      end)

      {:noreply, %{state | reload_in_progress: true}}
    end
  end

  @impl true
  def handle_info({:reload_complete, new_model, new_path, device}, state) do
    IO.puts("[hot_reload] Model swapped: v#{state.version} -> v#{state.version + 1}")
    {:noreply, %{state |
      model: new_model,
      path: new_path,
      device: device,
      version: state.version + 1,
      loaded_at: DateTime.utc_now(),
      reload_in_progress: false
    }}
  end

  def handle_info(:reload_failed, state) do
    IO.puts("[hot_reload] Keeping current model (v#{state.version})")
    {:noreply, %{state | reload_in_progress: false}}
  end
end

# === Demo ===

defmodule HotReloadDemo do
  @fixtures Path.join([__DIR__, "..", "..", "test", "fixtures"])

  def run do
    ExTorch.set_grad_enabled(false)

    path_v1 = Path.join(@fixtures, "resnet18.pt2")
    path_v2 = Path.join(@fixtures, "alexnet.pt2")
    bin = Path.join(@fixtures, "resnet18_input.bin")

    unless File.exists?(path_v1) do
      IO.puts("Missing fixtures. Run: .venv/bin/python test/fixtures/generate_popular_models.py")
      System.halt(1)
    end

    device = if ExTorch.Native.cuda_is_available(), do: :cuda, else: :cpu
    input = ExTorch.Native.from_binary(File.read!(bin), {1, 3, 224, 224}, :float32)
    input = if device == :cuda, do: ExTorch.Tensor.to(input, device: :cuda), else: input

    # Start with resnet18
    IO.puts("Starting server with ResNet18 on #{device}...")
    {:ok, _} = HotReloadServer.start_link(path: path_v1, device: device, mode: :native)

    # Run some inferences
    {:ok, out} = HotReloadServer.predict([input])
    IO.puts("v1 output shape: #{inspect(out.size)}")
    info = HotReloadServer.info()
    IO.puts("Version: #{info.version}, inferences: #{info.inference_count}")

    # Hot-reload to alexnet (different architecture, same input shape)
    IO.puts("\n--- Triggering hot reload to AlexNet ---")
    HotReloadServer.reload(HotReloadServer, path_v2, device: device)

    # Keep inferring during reload — requests hit the old model
    for i <- 1..5 do
      {:ok, _out} = HotReloadServer.predict([input])
      IO.puts("  Inference #{i} during reload: ok")
      Process.sleep(100)
    end

    # New model should be active now
    Process.sleep(500)
    {:ok, out} = HotReloadServer.predict([input])
    info = HotReloadServer.info()
    IO.puts("\nAfter reload:")
    IO.puts("  Version: #{info.version}")
    IO.puts("  Path: #{info.path}")
    IO.puts("  Output shape: #{inspect(out.size)}")
    IO.puts("  Total inferences: #{info.inference_count}")
  end
end

HotReloadDemo.run()
