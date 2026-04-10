# GPU inference latency benchmark for ExTorch.AOTI (compiled models).
#
# Usage:  mix run bench/inference_aoti_gpu.exs
#
# Loads each *_aoti.pt2 fixture on CUDA, runs inference N times, and reports
# min / median / mean latency. Compare against bench/inference_gpu.exs
# (Export interpreter) and bench/inference_gpu.py (PyTorch) to measure the
# benefit of AOT compilation vs per-op interpretation.

unless ExTorch.Native.cuda_is_available() do
  IO.puts("ERROR: CUDA is not available.")
  System.halt(1)
end

unless ExTorch.AOTI.available?() do
  IO.puts("ERROR: AOTI runtime not available in this libtorch build.")
  System.halt(1)
end

ExTorch.set_grad_enabled(false)

IO.puts("CUDA available: true")
IO.puts("AOTI available: true")
IO.puts("")

defmodule InferenceAotiGpu do
  @fixtures Path.join([__DIR__, "..", "test", "fixtures"])
  @warmup 5
  @iters 30

  # Tags: :production — real-world architectures
  #        :synthetic  — small models for micro-benchmarking
  @models [
    {"alexnet",            {1, 3, 224, 224},  [:production, :cnn]},
    {"vgg11",              {1, 3, 224, 224},  [:production, :cnn]},
    {"squeezenet",         {1, 3, 224, 224},  [:production, :cnn]},
    {"mobilenetv2",        {1, 3, 224, 224},  [:production, :cnn]},
    {"resnet18",           {1, 3, 224, 224},  [:production, :cnn]},
    {"resnet50",           {1, 3, 224, 224},  [:production, :cnn]},
    {"vit_b_16",           {1, 3, 224, 224},  [:production, :transformer]},
    {"simple_transformer", {1, 16, 32},       [:synthetic, :transformer]},
    {"mini_bert",          {1, 16, 32},       [:synthetic, :transformer]},
    {"autoencoder",        {1, 784},          [:synthetic, :mlp]},
    {"conv_autoencoder",   {1, 3, 32, 32},    [:synthetic, :cnn]},
  ]

  def run do
    IO.puts("== ExTorch.AOTI GPU latency  (#{@iters} iters after #{@warmup} warmup) ==\n")
    IO.puts(:io_lib.format(~c"~-22s ~10s ~10s ~10s", [~c"model", ~c"min(ms)", ~c"median(ms)", ~c"mean(ms)"]))
    IO.puts(String.duplicate("-", 56))

    for {name, in_shape, _tags} <- @models do
      aoti_path = Path.join(@fixtures, "#{name}_aoti.pt2")
      bin_path = Path.join(@fixtures, "#{name}_input.bin")

      if File.exists?(aoti_path) and File.exists?(bin_path) do
        model = ExTorch.AOTI.load(aoti_path, device_index: 0)
        cpu_input = ExTorch.Native.from_binary(File.read!(bin_path), in_shape, :float32)
        input = ExTorch.Tensor.to(cpu_input, device: :cuda)

        # Warmup
        for _ <- 1..@warmup, do: ExTorch.AOTI.forward(model, [input])
        ExTorch.cuda_synchronize()

        samples =
          for _ <- 1..@iters do
            {us, _} = :timer.tc(fn ->
              _out = ExTorch.AOTI.forward(model, [input])
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

InferenceAotiGpu.run()
