# Back-to-back A/B test: run the SAME benchmark twice in one process, once
# with the Phase D CPU fixes (no_grad + cleared affinity) and once without,
# so both measurements happen under the same thermal state.

defmodule AB do
  @fixtures Path.join([__DIR__, "..", "test", "fixtures"])
  @warmup 3
  @iters 20

  @models [
    {"alexnet",      {1, 3, 224, 224}},
    {"squeezenet",   {1, 3, 224, 224}},
    {"mobilenetv2",  {1, 3, 224, 224}},
    {"resnet18",     {1, 3, 224, 224}},
    {"resnet50",     {1, 3, 224, 224}},
    {"vgg11",        {1, 3, 224, 224}}
  ]

  def measure(label, fixes_applied) do
    IO.puts("\n== #{label} ==")
    IO.puts(:io_lib.format(~c"  ~-18s ~10s ~10s", [~c"model", ~c"median(ms)", ~c"min(ms)"]))
    IO.puts("  " <> String.duplicate("-", 44))
    for {name, in_shape} <- @models do
      model = ExTorch.Export.load(Path.join(@fixtures, "#{name}.pt2"))
      input = ExTorch.Native.from_binary(
        File.read!(Path.join(@fixtures, "#{name}_input.bin")),
        in_shape, :float32)
      # Warmup
      for _ <- 1..@warmup, do: ExTorch.Export.forward(model, [input])
      # Measure
      samples =
        for _ <- 1..@iters do
          {us, _} = :timer.tc(fn -> ExTorch.Export.forward(model, [input]) end)
          us / 1000.0
        end
      sorted = Enum.sort(samples)
      median = Enum.at(sorted, div(length(sorted), 2))
      min_v = List.first(sorted)
      IO.puts(:io_lib.format(~c"  ~-18s ~10.2f ~10.2f", [String.to_charlist(name), median, min_v]))
    end
    _ = fixes_applied
  end

  def run do
    # 1) Baseline: default state (grad on, original affinity)
    # (note: we can't un-clear affinity once cleared, so measure baseline FIRST)
    IO.puts("ExTorch initial state:")
    IO.puts("  grad_enabled=#{ExTorch.grad_enabled?()}")

    measure("Baseline (default: grad on, original affinity)", :none)

    # 2) With fixes applied
    ExTorch.set_grad_enabled(false)
    ExTorch.clear_cpu_affinity()

    measure("After fixes (no_grad + cleared affinity)", :both)
  end
end

AB.run()
