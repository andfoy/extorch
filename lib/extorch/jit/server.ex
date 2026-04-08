defmodule ExTorch.JIT.Server do
  @moduledoc """
  A GenServer that wraps a loaded TorchScript model for concurrent serving.

  Provides process isolation, fault tolerance, and serialized access to model
  inference. Forward calls are serialized through the GenServer to ensure
  thread safety for models with mutable state (e.g., BatchNorm, Dropout).

  ## Telemetry Events

  The server emits the following `:telemetry` events:

    * `[:extorch, :jit, :load, :start]` - When model loading begins.
      * Measurements: `%{system_time: integer}`
      * Metadata: `%{path: String.t(), device: atom()}`

    * `[:extorch, :jit, :load, :stop]` - When model loading completes.
      * Measurements: `%{duration: native_time}`
      * Metadata: `%{path: String.t(), device: atom()}`

    * `[:extorch, :jit, :load, :exception]` - When model loading fails.
      * Measurements: `%{duration: native_time}`
      * Metadata: `%{path: String.t(), device: atom(), kind: atom(), reason: term()}`

    * `[:extorch, :jit, :forward, :start]` - When inference begins.
      * Measurements: `%{system_time: integer}`
      * Metadata: `%{path: String.t(), device: atom(), input_count: integer()}`

    * `[:extorch, :jit, :forward, :stop]` - When inference completes.
      * Measurements: `%{duration: native_time}`
      * Metadata: `%{path: String.t(), device: atom(), input_count: integer()}`

    * `[:extorch, :jit, :forward, :exception]` - When inference fails.
      * Measurements: `%{duration: native_time}`
      * Metadata: `%{path: String.t(), device: atom(), input_count: integer(), kind: atom(), reason: term()}`

  ## Example

      {:ok, pid} = ExTorch.JIT.Server.start_link(path: "model.pt", device: :cpu)
      result = ExTorch.JIT.Server.predict(pid, [input_tensor])

  """

  use GenServer

  alias ExTorch.JIT

  defstruct [:model, :path, :device, :inference_count, :error_count, :started_at]

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start a model server.

  ## Options
    - `:path` (required) - Path to the `.pt` model file.
    - `:device` - Device to load the model onto (default: `:cpu`).
    - `:name` - Optional registered name for the server.
    - `:eval` - Whether to set the model to eval mode on load (default: `true`).
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    {name, opts} = Keyword.pop(opts, :name)
    server_opts = if name, do: [name: name], else: []
    GenServer.start_link(__MODULE__, opts, server_opts)
  end

  @doc """
  Run inference on the model (synchronous).

  ## Arguments
    - `server` - PID or registered name of the model server.
    - `inputs` - List of input tensors.
    - `timeout` - Call timeout in milliseconds (default: 30_000).

  ## Returns
  The model output.
  """
  @spec predict(GenServer.server(), [ExTorch.Tensor.t()], timeout()) :: term()
  def predict(server, inputs, timeout \\ 30_000) when is_list(inputs) do
    GenServer.call(server, {:predict, inputs}, timeout)
  end

  @doc """
  Get information about the loaded model.
  """
  @spec info(GenServer.server()) :: map()
  def info(server) do
    GenServer.call(server, :info)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    path = Keyword.fetch!(opts, :path)
    device = Keyword.get(opts, :device, :cpu)
    eval_mode = Keyword.get(opts, :eval, true)
    metadata = %{path: path, device: device}

    model =
      :telemetry.span([:extorch, :jit, :load], metadata, fn ->
        m = JIT.load(path, device: device)
        {m, metadata}
      end)

    if eval_mode do
      JIT.eval(model)
    end

    state = %__MODULE__{
      model: model,
      path: path,
      device: device,
      inference_count: 0,
      error_count: 0,
      started_at: System.monotonic_time(:millisecond)
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:predict, inputs}, _from, state) do
    metadata = %{
      path: state.path,
      device: state.device,
      input_count: length(inputs)
    }

    try do
      result =
        :telemetry.span([:extorch, :jit, :forward], metadata, fn ->
          r = JIT.forward(state.model, inputs)
          {r, metadata}
        end)

      new_state = %{state | inference_count: state.inference_count + 1}
      {:reply, result, new_state}
    rescue
      e ->
        :telemetry.execute(
          [:extorch, :jit, :forward, :exception],
          %{system_time: System.system_time()},
          Map.merge(metadata, %{kind: :error, reason: e})
        )

        new_state = %{state | error_count: state.error_count + 1}
        {:reply, {:error, e}, new_state}
    end
  end

  @impl true
  def handle_call(:info, _from, state) do
    uptime_ms = System.monotonic_time(:millisecond) - state.started_at

    info = %{
      path: state.path,
      device: state.device,
      inference_count: state.inference_count,
      error_count: state.error_count,
      uptime_ms: uptime_ms
    }

    {:reply, info, state}
  end

  @impl true
  def format_status(_reason, [_pdict, state]) do
    %{
      path: state.path,
      device: state.device,
      inference_count: state.inference_count,
      error_count: state.error_count,
      uptime_ms: System.monotonic_time(:millisecond) - state.started_at
    }
  end
end
