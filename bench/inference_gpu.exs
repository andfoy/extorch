# GPU inference latency benchmark for ExTorch.Export.
#
# Usage:  mix run bench/inference_gpu.exs
#
# Loads each popular_models fixture on :cuda, runs the saved input on :cuda
# through ExTorch.Export.forward N times, and reports min / median / mean
# latency. Requires a CUDA-capable libtorch build (check via
# `ExTorch.cuda_is_available/0`).

unless ExTorch.Native.cuda_is_available() do
  IO.puts("ERROR: CUDA is not available in this libtorch build.")
  IO.puts("Make sure priv/native/libtorch is a CUDA build (2.11.0+cuXYZ).")
  System.halt(1)
end

# Inference-mode fixes: disable autograd graph build-up process-wide.
# clear_cpu_affinity is harmless on GPU (useful work happens on the GPU,
# not on CPU threads) but costs nothing.
ExTorch.set_grad_enabled(false)
ExTorch.clear_cpu_affinity()

IO.puts("ExTorch backend: #{ExTorch.Native.aten_backend_info()}")
IO.puts("CUDA available:  true")
IO.puts("")

defmodule InferenceGpu do
  @fixtures Path.join([__DIR__, "..", "test", "fixtures"])
  @warmup 5
  @iters 30

  @models [
    {"alexnet",            {1, 3, 224, 224},  {1, 1000}},
    {"vgg11",              {1, 3, 224, 224},  {1, 1000}},
    {"squeezenet",         {1, 3, 224, 224},  {1, 1000}},
    {"mobilenetv2",        {1, 3, 224, 224},  {1, 1000}},
    {"resnet18",           {1, 3, 224, 224},  {1, 1000}},
    {"resnet50",           {1, 3, 224, 224},  {1, 1000}},
    {"simple_transformer", {1, 16, 32},       {1, 10}},
    {"mini_bert",          {1, 16, 32},       {1, 10}},
    {"autoencoder",        {1, 784},          {1, 784}},
    {"conv_autoencoder",   {1, 3, 32, 32},    {1, 3, 32, 32}},
    {"simple_lstm",        {1, 16, 32},       {1, 10}}
  ]

  def run do
    IO.puts("== ExTorch.Export GPU latency  (#{@iters} iters after #{@warmup} warmup) ==\n")
    IO.puts(:io_lib.format(~c"~-22s ~10s ~10s ~10s", [~c"model", ~c"min(ms)", ~c"median(ms)", ~c"mean(ms)"]))
    IO.puts(String.duplicate("-", 56))

    for {name, in_shape, _out_shape} <- @models do
      path = Path.join(@fixtures, "#{name}.pt2")
      bin = Path.join(@fixtures, "#{name}_input.bin")
      if File.exists?(path) and File.exists?(bin) do
        # Load model with weights placed on CUDA up front.
        model = ExTorch.Export.load(path, device: :cuda)
        # Load input on CPU then move to CUDA.
        cpu_input = ExTorch.Native.from_binary(File.read!(bin), in_shape, :float32)
        input = ExTorch.Tensor.to(cpu_input, device: :cuda)

        # Warmup — first CUDA calls compile/select kernels and allocate
        # the CUDA caching allocator pool; we need more warmups than CPU.
        for _ <- 1..@warmup, do: ExTorch.Export.forward(model, [input])
        ExTorch.cuda_synchronize()

        samples =
          for _ <- 1..@iters do
            # Sync before AND after so the measured interval only
            # contains this forward's kernels. CUDA launches are async.
            {us, _} = :timer.tc(fn ->
              _out = ExTorch.Export.forward(model, [input])
              ExTorch.cuda_synchronize()
            end)
            us / 1000.0
          end

        sorted = Enum.sort(samples)
        min_v = List.first(sorted)
        median = Enum.at(sorted, div(length(sorted), 2))
        mean = Enum.sum(samples) / length(samples)
        IO.puts(:io_lib.format(~c"~-22s ~10.2f ~10.2f ~10.2f", [String.to_charlist(name), min_v, median, mean]))
      else
        IO.puts(:io_lib.format(~c"~-22s    SKIP (fixture missing)", [String.to_charlist(name)]))
      end
    end
  end
end

InferenceGpu.run()
