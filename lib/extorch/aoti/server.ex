defmodule ExTorch.AOTI.Server do
  @moduledoc """
  A GenServer that wraps a loaded AOTI (.pt2) model for concurrent serving.

  Provides the same OTP fault tolerance and telemetry instrumentation as
  `ExTorch.JIT.Server`, but for AOTInductor-compiled models.

  ## Telemetry Events

  The server emits the following `:telemetry` events:

    * `[:extorch, :aoti, :load, :start | :stop]` - Model loading.
    * `[:extorch, :aoti, :forward, :start | :stop | :exception]` - Inference.

  All events include `%{path: String.t()}` in metadata. Forward events also
  include `%{input_count: integer()}`.

  ## Example

      {:ok, pid} = ExTorch.AOTI.Server.start_link(path: "model.pt2")
      [output] = ExTorch.AOTI.Server.predict(pid, [input])

  ## Named servers

      {:ok, _} = ExTorch.AOTI.Server.start_link(path: "model.pt2", name: FastModel)
      [output] = ExTorch.AOTI.Server.predict(FastModel, [input])

  """

  use GenServer

  alias ExTorch.AOTI

  defstruct [:model, :path, :inference_count, :error_count, :started_at]

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start an AOTI model server.

  ## Options
    - `:path` (required) - Path to the `.pt2` model package.
    - `:name` - Optional registered name.
    - `:model_name` - Model name within the package (default: `"model"`).
    - `:device_index` - CUDA device index (default: `-1` for CPU).
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    {name, opts} = Keyword.pop(opts, :name)
    server_opts = if name, do: [name: name], else: []
    GenServer.start_link(__MODULE__, opts, server_opts)
  end

  @doc """
  Run inference on the model (synchronous).

  ## Returns
  A list of output tensors.
  """
  @spec predict(GenServer.server(), [ExTorch.Tensor.t()], timeout()) :: [ExTorch.Tensor.t()]
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
    model_name = Keyword.get(opts, :model_name, "model")
    device_index = Keyword.get(opts, :device_index, -1)
    metadata = %{path: path}

    model =
      :telemetry.span([:extorch, :aoti, :load], metadata, fn ->
        m = AOTI.load(path, model_name: model_name, device_index: device_index)
        {m, metadata}
      end)

    state = %__MODULE__{
      model: model,
      path: path,
      inference_count: 0,
      error_count: 0,
      started_at: System.monotonic_time(:millisecond)
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:predict, inputs}, _from, state) do
    metadata = %{path: state.path, input_count: length(inputs)}

    try do
      result =
        :telemetry.span([:extorch, :aoti, :forward], metadata, fn ->
          r = AOTI.forward(state.model, inputs)
          {r, metadata}
        end)

      new_state = %{state | inference_count: state.inference_count + 1}
      {:reply, result, new_state}
    rescue
      e ->
        :telemetry.execute(
          [:extorch, :aoti, :forward, :exception],
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
      inference_count: state.inference_count,
      error_count: state.error_count,
      uptime_ms: System.monotonic_time(:millisecond) - state.started_at
    }
  end
end
