# Basic Inference — Three Paths
#
# Demonstrates ExTorch's three inference paths and when to use each.
#
# Usage: mix run examples/serving/basic_inference.exs
#
# Requirements: run test/fixtures/generate_popular_models.py first

defmodule BasicInference do
  @fixtures Path.join([__DIR__, "..", "..", "test", "fixtures"])

  def run do
    ExTorch.set_grad_enabled(false)

    model_name = "resnet18"
    pt2_path = Path.join(@fixtures, "#{model_name}.pt2")
    input_path = Path.join(@fixtures, "#{model_name}_input.bin")

    unless File.exists?(pt2_path) do
      IO.puts("Missing #{pt2_path}. Run: .venv/bin/python test/fixtures/generate_popular_models.py")
      System.halt(1)
    end

    input = ExTorch.Native.from_binary(File.read!(input_path), {1, 3, 224, 224}, :float32)
    device = if ExTorch.Native.cuda_is_available(), do: :cuda, else: :cpu
    IO.puts("Device: #{device}\n")

    # =========================================================================
    # Path 1: Export Interpreter (forward/2)
    #
    # Best for: development, debugging, op-by-op profiling
    # Trade-off: per-node NIF overhead on high-node models (ViT)
    # =========================================================================
    IO.puts("=== Path 1: Export Interpreter ===")
    model = ExTorch.Export.load(pt2_path, device: device)
    model_input = if device == :cuda, do: ExTorch.Tensor.to(input, device: :cuda), else: input

    {us, output} = :timer.tc(fn -> ExTorch.Export.forward(model, [model_input]) end)
    IO.puts("  Output shape: #{inspect(output.size)}")
    IO.puts("  Latency: #{Float.round(us / 1000, 2)}ms")
    IO.puts("  Use case: development, debugging, profiling\n")

    # =========================================================================
    # Path 2: Export Native (forward_native/2)
    #
    # Best for: production serving without pre-compilation
    # Runs entire graph in a single NIF call via C++ executor
    # =========================================================================
    IO.puts("=== Path 2: Export Native (C++ graph executor) ===")

    {us, output} = :timer.tc(fn -> ExTorch.Export.forward_native(model, [model_input]) end)
    IO.puts("  Output shape: #{inspect(output.size)}")
    IO.puts("  Latency: #{Float.round(us / 1000, 2)}ms")
    IO.puts("  Use case: production serving, best for transformers\n")

    # =========================================================================
    # Path 3: AOTI (compiled inference)
    #
    # Best for: maximum throughput, fused kernels
    # Requires pre-compilation via torch._inductor.aoti_compile_and_package
    # =========================================================================
    aoti_path = Path.join(@fixtures, "#{model_name}_aoti.pt2")

    if File.exists?(aoti_path) and ExTorch.AOTI.available?() do
      IO.puts("=== Path 3: AOTI (compiled kernels) ===")
      aoti_model = ExTorch.AOTI.load(aoti_path, device_index: if(device == :cuda, do: 0, else: -1))

      {us, [output]} = :timer.tc(fn -> ExTorch.AOTI.forward(aoti_model, [model_input]) end)
      IO.puts("  Output shape: #{inspect(output.size)}")
      IO.puts("  Latency: #{Float.round(us / 1000, 2)}ms")
      IO.puts("  Use case: maximum throughput, pre-compiled models\n")
    else
      IO.puts("=== Path 3: AOTI (skipped — run generate_aoti_popular_models.py) ===\n")
    end

    # =========================================================================
    # Benchmark comparison
    # =========================================================================
    IO.puts("=== Latency Comparison (30 iterations, #{device}) ===")
    iters = 30

    # Warmup
    for _ <- 1..5 do
      ExTorch.Export.forward(model, [model_input])
      ExTorch.Export.forward_native(model, [model_input])
    end
    if device == :cuda, do: ExTorch.cuda_synchronize()

    interp_samples = for _ <- 1..iters do
      {us, _} = :timer.tc(fn ->
        ExTorch.Export.forward(model, [model_input])
        if device == :cuda, do: ExTorch.cuda_synchronize()
      end)
      us / 1000.0
    end

    native_samples = for _ <- 1..iters do
      {us, _} = :timer.tc(fn ->
        ExTorch.Export.forward_native(model, [model_input])
        if device == :cuda, do: ExTorch.cuda_synchronize()
      end)
      us / 1000.0
    end

    interp_med = Enum.sort(interp_samples) |> Enum.at(div(iters, 2))
    native_med = Enum.sort(native_samples) |> Enum.at(div(iters, 2))

    IO.puts("  Interpreter: #{Float.round(interp_med, 2)}ms (median)")
    IO.puts("  Native:      #{Float.round(native_med, 2)}ms (median)")
    IO.puts("  Speedup:     #{Float.round(interp_med / native_med, 2)}x")
  end
end

BasicInference.run()
