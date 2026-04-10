# Same as inference.exs but applies the two newly-discovered CPU fixes
# before running: disable grad mode globally and clear the current thread's
# CPU affinity mask so libtorch's OpenMP workers can use all cores.

ExTorch.Native.aten_set_grad_enabled(false)
ExTorch.Native.aten_clear_cpu_affinity()

defmodule InferenceFast do
  @fixtures Path.join([__DIR__, "..", "test", "fixtures"])
  @warmup 3
  @iters 30

  # Tags: :production — real-world architectures used in production serving
  #        :synthetic  — small custom models for micro-benchmarking dispatch overhead
  @models [
    # Production vision models
    {"alexnet",            {1, 3, 224, 224},  {1, 1000},       [:production, :cnn]},
    {"vgg11",              {1, 3, 224, 224},  {1, 1000},       [:production, :cnn]},
    {"squeezenet",         {1, 3, 224, 224},  {1, 1000},       [:production, :cnn]},
    {"mobilenetv2",        {1, 3, 224, 224},  {1, 1000},       [:production, :cnn]},
    {"resnet18",           {1, 3, 224, 224},  {1, 1000},       [:production, :cnn]},
    {"resnet50",           {1, 3, 224, 224},  {1, 1000},       [:production, :cnn]},
    {"vit_b_16",           {1, 3, 224, 224},  {1, 1000},       [:production, :transformer]},
    # Synthetic / micro-benchmark models
    {"simple_transformer", {1, 16, 32},       {1, 10},         [:synthetic, :transformer]},
    {"mini_bert",          {1, 16, 32},       {1, 10},         [:synthetic, :transformer]},
    {"autoencoder",        {1, 784},          {1, 784},        [:synthetic, :mlp]},
    {"conv_autoencoder",   {1, 3, 32, 32},    {1, 3, 32, 32},  [:synthetic, :cnn]},
    {"simple_lstm",        {1, 16, 32},       {1, 10},         [:synthetic, :rnn]}
  ]

  def run do
    IO.puts("== ExTorch.Export (grad_disabled + cleared affinity)  #{@iters} iters ==\n")
    IO.puts(:io_lib.format(~c"~-22s ~10s ~10s ~10s", [~c"model", ~c"min(ms)", ~c"median(ms)", ~c"mean(ms)"]))
    IO.puts(String.duplicate("-", 56))

    for {name, in_shape, _out_shape, _tags} <- @models do
      path = Path.join(@fixtures, "#{name}.pt2")
      bin = Path.join(@fixtures, "#{name}_input.bin")
      if File.exists?(path) and File.exists?(bin) do
        model = ExTorch.Export.load(path)
        input = ExTorch.Native.from_binary(File.read!(bin), in_shape, :float32)
        for _ <- 1..@warmup, do: ExTorch.Export.forward(model, [input])
        samples =
          for _ <- 1..@iters do
            {us, _} = :timer.tc(fn -> ExTorch.Export.forward(model, [input]) end)
            us / 1000.0
          end
        sorted = Enum.sort(samples)
        min_v = List.first(sorted)
        median = Enum.at(sorted, div(length(sorted), 2))
        mean = Enum.sum(samples) / length(samples)
        IO.puts(:io_lib.format(~c"~-22s ~10.2f ~10.2f ~10.2f", [String.to_charlist(name), min_v, median, mean]))
      end
    end
  end
end

InferenceFast.run()
