defmodule ExTorch.Export.OpRegistry do
  @moduledoc """
  Registry for custom `ExTorch.Export.OpHandler` modules.

  Maps op target strings to handler modules so `ExTorch.Export` can dispatch
  to external op implementations at both compile time (load) and runtime
  (forward).

  Handlers are registered in two ways:

  1. **Config** (loaded at first access):

      config :extorch, :export_op_handlers, [MyApp.VisionOps]

  2. **Runtime**:

      ExTorch.Export.OpRegistry.register(MyApp.VisionOps)

  The registry is backed by a persistent_term for fast concurrent reads.
  """

  @registry_key {__MODULE__, :dispatch_table}

  @doc """
  Register an `ExTorch.Export.OpHandler` module.

  Merges the handler's ops into the dispatch table. If an op is already
  registered by another handler, the new handler takes precedence.
  """
  @spec register(module()) :: :ok
  def register(handler) when is_atom(handler) do
    new_entries =
      handler.ops()
      |> Enum.map(fn op -> {op, handler} end)
      |> Map.new()

    current = get_table()
    :persistent_term.put(@registry_key, Map.merge(current, new_entries))
    :ok
  end

  @doc """
  Look up the handler module for a given op target string.

  Returns `{:ok, handler_module}` or `:error`.
  """
  @spec lookup(String.t()) :: {:ok, module()} | :error
  def lookup(op_target) do
    Map.fetch(get_table(), op_target)
  end

  @doc """
  Returns the full dispatch table as a `%{op_target => handler_module}` map.
  """
  @spec table() :: %{String.t() => module()}
  def table, do: get_table()

  @doc """
  Remove all registered handlers and reset the dispatch table.
  """
  @spec reset() :: :ok
  def reset do
    :persistent_term.put(@registry_key, %{})
    :ok
  end

  # Lazily initialize from application config on first access.
  defp get_table do
    case :persistent_term.get(@registry_key, :not_loaded) do
      :not_loaded ->
        handlers = Application.get_env(:extorch, :export_op_handlers, [])
        table =
          Enum.reduce(handlers, %{}, fn handler, acc ->
            entries = handler.ops() |> Enum.map(&{&1, handler}) |> Map.new()
            Map.merge(acc, entries)
          end)
        :persistent_term.put(@registry_key, table)
        table

      table ->
        table
    end
  end
end
