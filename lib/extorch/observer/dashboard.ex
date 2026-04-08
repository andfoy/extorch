if Code.ensure_loaded?(Phoenix.LiveDashboard.PageBuilder) do
  defmodule ExTorch.Observer.Dashboard do
    @moduledoc """
    A Phoenix LiveDashboard page for monitoring ExTorch model serving.

    ## Setup

    Add to your LiveDashboard configuration:

        live_dashboard "/dashboard",
          additional_pages: [
            extorch: ExTorch.Observer.Dashboard
          ]

    Make sure `ExTorch.Metrics.setup/0` is called in your application start.
    """

    use Phoenix.LiveDashboard.PageBuilder

    @impl true
    def menu_link(_, _) do
      {:ok, "ExTorch"}
    end

    @impl true
    def render(assigns) do
      metrics = ExTorch.Metrics.all()
      cuda_available = ExTorch.Native.cuda_is_available()

      rows =
        Enum.map(metrics, fn {path, m} ->
          avg_ms =
            if m.inference_count > 0,
              do: Float.round(m.total_duration_ms / m.inference_count, 2),
              else: 0.0

          %{
            path: Path.basename(path),
            device: to_string(m.device),
            inferences: m.inference_count,
            errors: m.error_count,
            avg_ms: avg_ms,
            min_ms:
              if(m.min_duration_ms == :infinity, do: "-", else: "#{Float.round(m.min_duration_ms, 2)}"),
            max_ms: "#{Float.round(m.max_duration_ms, 2)}",
            load_ms: "#{Float.round(m.load_duration_ms, 2)}"
          }
        end)

      assigns =
        assigns
        |> Map.put(:rows, rows)
        |> Map.put(:cuda_available, cuda_available)

      ~H"""
      <h5>ExTorch Model Serving</h5>
      <p>CUDA: <%= if @cuda_available, do: "Available", else: "Not available" %></p>

      <%= if @rows == [] do %>
        <p>No models loaded. Start an ExTorch.JIT.Server to see metrics.</p>
      <% else %>
        <table>
          <thead>
            <tr>
              <th>Model</th><th>Device</th><th>Inferences</th>
              <th>Errors</th><th>Avg (ms)</th><th>Min (ms)</th>
              <th>Max (ms)</th><th>Load (ms)</th>
            </tr>
          </thead>
          <tbody>
            <%= for row <- @rows do %>
              <tr>
                <td><%= row.path %></td>
                <td><%= row.device %></td>
                <td><%= row.inferences %></td>
                <td><%= row.errors %></td>
                <td><%= row.avg_ms %></td>
                <td><%= row.min_ms %></td>
                <td><%= row.max_ms %></td>
                <td><%= row.load_ms %></td>
              </tr>
            <% end %>
          </tbody>
        </table>
      <% end %>
      """
    end
  end
end
