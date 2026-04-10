# Microbenchmark: where does the per-op overhead in the interpreter live?
#
# Compares three call patterns on a single conv2d:
#   A. construction + copy_parameters + forward  (what execute_conv2d does today)
#   B. forward through a pre-built layer with weights already copied
#   C. construction only (no copy, no forward) - to isolate layer alloc cost
#
# A - B is the per-call overhead the interpreter pays. If that's a large
# fraction of A, migrating conv to direct at::conv2d eliminates it.

defmodule LayerOverhead do
  @iters 1000

  def run do
    in_ch = 64
    out_ch = 128
    k = 3
    h = 56
    w = 56

    input = ExTorch.randn({1, in_ch, h, w})
    weight = ExTorch.randn({out_ch, in_ch, k, k})
    bias = ExTorch.randn({out_ch})

    # Warm up the JIT/cache
    _ = ExTorch.NN.conv2d(in_ch, out_ch, k, padding: 1)
    prebuilt = ExTorch.NN.conv2d(in_ch, out_ch, k, padding: 1)
    ExTorch.NN.copy_parameters(prebuilt, [{"weight", weight}, {"bias", bias}])
    _ = ExTorch.NN.forward(input, prebuilt)

    # A: full cycle (current execute_conv2d behaviour)
    {us_a, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        layer = ExTorch.NN.conv2d(in_ch, out_ch, k, padding: 1)
        ExTorch.NN.copy_parameters(layer, [{"weight", weight}, {"bias", bias}])
        ExTorch.NN.forward(input, layer)
      end
    end)

    # B: forward only (the irreducible work)
    {us_b, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        ExTorch.NN.forward(input, prebuilt)
      end
    end)

    # C: construction only
    {us_c, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        ExTorch.NN.conv2d(in_ch, out_ch, k, padding: 1)
      end
    end)

    # D: construction + copy_parameters (no forward)
    {us_d, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        layer = ExTorch.NN.conv2d(in_ch, out_ch, k, padding: 1)
        ExTorch.NN.copy_parameters(layer, [{"weight", weight}, {"bias", bias}])
      end
    end)

    a_per = us_a / @iters
    b_per = us_b / @iters
    c_per = us_c / @iters
    d_per = us_d / @iters

    IO.puts("\n== conv2d microbenchmark  (#{@iters} iters, shape=#{inspect({1, in_ch, h, w})}, k=#{k}) ==\n")
    IO.puts("  A. construct + copy_parameters + forward : #{:io_lib.format(~c"~7.1f", [a_per])} us")
    IO.puts("  B. forward only (pre-built layer)        : #{:io_lib.format(~c"~7.1f", [b_per])} us")
    IO.puts("  C. construct only                        : #{:io_lib.format(~c"~7.1f", [c_per])} us")
    IO.puts("  D. construct + copy_parameters           : #{:io_lib.format(~c"~7.1f", [d_per])} us")
    IO.puts("")
    overhead = a_per - b_per
    pct = overhead / a_per * 100
    IO.puts("  Per-call overhead (A - B) : #{:io_lib.format(~c"~7.1f", [overhead])} us  (#{:io_lib.format(~c"~5.1f", [pct])}% of A)")
    speedup = a_per / b_per
    IO.puts("  Forward-only speedup (A / B): #{:io_lib.format(~c"~5.2f", [speedup])}x")
  end
end

LayerOverhead.run()
