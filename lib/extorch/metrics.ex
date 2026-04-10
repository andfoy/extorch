defmodule ExTorch.Metrics do
  @moduledoc """
  ETS-backed metrics collection for ExTorch model serving.

  Automatically attaches to telemetry events emitted by `ExTorch.JIT.Server`
  and maintains per-model inference statistics.

  ## Setup

  Call `ExTorch.Metrics.setup/0` in your application start to begin collecting:

      def start(_type, _args) do
        ExTorch.Metrics.setup()
        # ...
      end

  ## Querying Metrics

      ExTorch.Metrics.get("model.pt")
      # => %{inference_count: 150, error_count: 2, total_duration_ms: 4523.1,
      #      min_duration_ms: 12.3, max_duration_ms: 89.2, last_inference_at: ~U[...]}

      ExTorch.Metrics.all()
      # => [{"model.pt", %{...}}, {"other.pt", %{...}}]

  """

  @table __MODULE__
  @handler_id "extorch-metrics"

  @doc """
  Initialize the metrics ETS table and attach telemetry handlers.

  Safe to call multiple times -- will not reset existing data.
  """
  @spec setup() :: :ok
  def setup do
    if :ets.whereis(@table) == :undefined do
      :ets.new(@table, [:named_table, :public, :set, read_concurrency: true])
    end

    events = [
      [:extorch, :jit, :forward, :stop],
      [:extorch, :jit, :forward, :exception],
      [:extorch, :jit, :load, :stop]
    ]

    :telemetry.attach_many(@handler_id, events, &handle_event/4, nil)
    :ok
  end

  @doc """
  Detach telemetry handlers. Metrics table remains intact.
  """
  @spec teardown() :: :ok | {:error, :not_found}
  def teardown do
    :telemetry.detach(@handler_id)
  end

  @doc """
  Get metrics for a specific model path.

  Returns a map of metrics or `nil` if no data exists.
  """
  @spec get(String.t()) :: map() | nil
  def get(path) do
    case :ets.lookup(@table, path) do
      [{^path, metrics}] -> metrics
      [] -> nil
    end
  end

  @doc """
  Get metrics for all tracked models.

  Returns a list of `{path, metrics}` tuples.
  """
  @spec all() :: [{String.t(), map()}]
  def all do
    :ets.tab2list(@table)
  end

  @doc """
  Reset metrics for a specific model path.
  """
  @spec reset(String.t()) :: :ok
  def reset(path) do
    :ets.delete(@table, path)
    :ok
  end

  @doc """
  Reset all metrics.
  """
  @spec reset_all() :: :ok
  def reset_all do
    :ets.delete_all_objects(@table)
    :ok
  end

  # ============================================================================
  # Telemetry handlers
  # ============================================================================

  @doc false
  def handle_event([:extorch, :jit, :forward, :stop], measurements, metadata, _config) do
    path = metadata.path
    duration_ms = System.convert_time_unit(measurements.duration, :native, :microsecond) / 1000

    update_metrics(path, fn metrics ->
      count = metrics.inference_count + 1

      %{
        metrics
        | inference_count: count,
          total_duration_ms: metrics.total_duration_ms + duration_ms,
          min_duration_ms: min(metrics.min_duration_ms, duration_ms),
          max_duration_ms: max(metrics.max_duration_ms, duration_ms),
          last_inference_at: System.system_time(:millisecond)
      }
    end)
  end

  def handle_event([:extorch, :jit, :forward, :exception], _measurements, metadata, _config) do
    update_metrics(metadata.path, fn metrics ->
      %{metrics | error_count: metrics.error_count + 1}
    end)
  end

  def handle_event([:extorch, :jit, :load, :stop], measurements, metadata, _config) do
    path = metadata.path
    duration_ms = System.convert_time_unit(measurements.duration, :native, :microsecond) / 1000

    update_metrics(path, fn metrics ->
      %{metrics | load_duration_ms: duration_ms, device: metadata.device}
    end)
  end

  defp update_metrics(path, update_fn) do
    # Guard against the ETS table not existing. The table is created by
    # setup/0 and owned by its caller; if that process exits the table is
    # destroyed, but the telemetry handler persists. Silently skip the
    # update rather than crashing the handler with :badarg.
    if :ets.whereis(@table) == :undefined, do: :ok, else: do_update_metrics(path, update_fn)
  end

  defp do_update_metrics(path, update_fn) do
    default = %{
      inference_count: 0,
      error_count: 0,
      total_duration_ms: 0.0,
      min_duration_ms: :infinity,
      max_duration_ms: 0.0,
      load_duration_ms: 0.0,
      device: :cpu,
      last_inference_at: nil
    }

    current =
      case :ets.lookup(@table, path) do
        [{^path, metrics}] -> metrics
        [] -> default
      end

    :ets.insert(@table, {path, update_fn.(current)})
  end
end
