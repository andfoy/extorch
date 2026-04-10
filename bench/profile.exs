# Per-node profiling: which ops dominate inference time?

defmodule ProfileBench do
  @fixtures Path.join([__DIR__, "..", "test", "fixtures"])
  @warmup 3
  @iters 10

  def run(name, in_shape) do
    model = ExTorch.Export.load(Path.join(@fixtures, "#{name}.pt2"))
    input = ExTorch.Native.from_binary(
      File.read!(Path.join(@fixtures, "#{name}_input.bin")),
      in_shape, :float32)

    # Warm up
    for _ <- 1..@warmup, do: ExTorch.Export.forward(model, [input])

    # Aggregate stats across @iters runs
    stats =
      for _ <- 1..@iters do
        {_out, s} = ExTorch.Export.forward_profiled(model, [input])
        s
      end
      |> Enum.reduce(%{}, fn run_stats, acc ->
        Enum.reduce(run_stats, acc, fn {target, %{count: c, total_ns: ns}}, m ->
          Map.update(m, target, %{count: c, total_ns: ns}, fn e ->
            %{count: e.count + c, total_ns: e.total_ns + ns}
          end)
        end)
      end)

    # Also measure total forward time for context
    {total_us, _} = :timer.tc(fn ->
      for _ <- 1..@iters, do: ExTorch.Export.forward(model, [input])
    end)
    total_ms = total_us / @iters / 1000
    per_node_total_ms =
      stats
      |> Map.values()
      |> Enum.reduce(0, fn e, acc -> acc + e.total_ns end)
      |> then(&(&1 / @iters / 1_000_000))

    IO.puts("\n== #{name}  (avg over #{@iters} iters) ==")
    IO.puts("  total forward time: #{:io_lib.format(~c"~7.2f", [total_ms])} ms")
    IO.puts("  sum of per-node:    #{:io_lib.format(~c"~7.2f", [per_node_total_ms])} ms  (#{:io_lib.format(~c"~4.1f", [per_node_total_ms / total_ms * 100])}%)")
    IO.puts("")
    IO.puts(:io_lib.format(~c"  ~-55s ~7s ~10s ~7s", [~c"op", ~c"count", ~c"total(ms)", ~c"%"]))
    IO.puts("  " <> String.duplicate("-", 85))

    sorted =
      stats
      |> Enum.map(fn {target, %{count: c, total_ns: ns}} ->
        {target, c, ns / @iters / 1_000_000}
      end)
      |> Enum.sort_by(fn {_, _, ms} -> ms end, :desc)

    Enum.each(sorted, fn {target, count, ms} ->
      short = String.replace(target, "torch.ops.aten.", "")
      pct = ms / total_ms * 100
      IO.puts(:io_lib.format(~c"  ~-55s ~7B ~10.2f ~6.1f%", [String.to_charlist(short), div(count, @iters), ms, pct]))
    end)
  end
end

ProfileBench.run("resnet50", {1, 3, 224, 224})
ProfileBench.run("mobilenetv2", {1, 3, 224, 224})
