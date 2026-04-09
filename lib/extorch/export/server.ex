defmodule ExTorch.Export.Server do
  @moduledoc """
  A GenServer that wraps a loaded `torch.export.save` model for concurrent serving.

  Uses the pure Elixir ATen graph interpreter -- no JIT, no AOTI, no C++
  ExportedProgram support needed.

  ## Telemetry Events

  The server emits the following `:telemetry` events:

    * `[:extorch, :export, :load, :start | :stop]` - Model loading.
    * `[:extorch, :export, :forward, :start | :stop | :exception]` - Inference.

  All events include `%{path: String.t()}` in metadata. Forward events also
  include `%{input_count: integer()}`.

  ## Example

      {:ok, pid} = ExTorch.Export.Server.start_link(path: "model.pt2")
      output = ExTorch.Export.Server.predict(pid, [input])

  ## Named servers

      {:ok, _} = ExTorch.Export.Server.start_link(path: "model.pt2", name: MyModel)
      output = ExTorch.Export.Server.predict(MyModel, [input])

  """

  use GenServer

  alias ExTorch.Export

  defstruct [:model, :path, :inference_count, :error_count, :started_at]

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Start an Export model server.

  ## Options
    - `:path` (required) - Path to the `.pt2` archive from `torch.export.save`.
    - `:name` - Optional registered name.
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
  The output tensor (or list of tensors for multi-output models).
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
    metadata = %{path: path}

    model =
      :telemetry.span([:extorch, :export, :load], metadata, fn ->
        m = Export.load(path)
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
        :telemetry.span([:extorch, :export, :forward], metadata, fn ->
          r = Export.forward(state.model, inputs)
          {r, metadata}
        end)

      new_state = %{state | inference_count: state.inference_count + 1}
      {:reply, result, new_state}
    rescue
      e ->
        :telemetry.execute(
          [:extorch, :export, :forward, :exception],
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
