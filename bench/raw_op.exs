# Microbenchmark: pure NIF call cost for a single conv2d, no interpreter loop.
# Use this to find out whether the remaining CNN gap is in NIF marshaling
# overhead or in the actual at::conv2d kernel time.

defmodule RawOp do
  @iters 200

  def run do
    # ResNet50's first conv: 3->64, kernel=7, stride=2, padding=3
    input = ExTorch.randn({1, 3, 224, 224})
    weight = ExTorch.randn({64, 3, 7, 7})

    # Warm up
    for _ <- 1..3, do: ExTorch.Native.aten_conv2d(input, weight, nil, [2, 2], [3, 3], [1, 1], 1)

    {us, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        ExTorch.Native.aten_conv2d(input, weight, nil, [2, 2], [3, 3], [1, 1], 1)
      end
    end)
    IO.puts("conv2d 3->64 k=7 s=2 p=3 input=224x224: #{:io_lib.format(~c"~7.2f", [us / @iters / 1000])} ms/call")

    # ResNet18 layer1 conv: 64->64 k=3 s=1 p=1
    inp2 = ExTorch.randn({1, 64, 56, 56})
    w2 = ExTorch.randn({64, 64, 3, 3})
    for _ <- 1..3, do: ExTorch.Native.aten_conv2d(inp2, w2, nil, [1, 1], [1, 1], [1, 1], 1)
    {us, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        ExTorch.Native.aten_conv2d(inp2, w2, nil, [1, 1], [1, 1], [1, 1], 1)
      end
    end)
    IO.puts("conv2d 64->64 k=3 s=1 p=1 input=56x56:    #{:io_lib.format(~c"~7.2f", [us / @iters / 1000])} ms/call")

    # ResNet50 deeper conv: 256->64 k=1 s=1 p=0 (bottleneck)
    inp3 = ExTorch.randn({1, 256, 56, 56})
    w3 = ExTorch.randn({64, 256, 1, 1})
    for _ <- 1..3, do: ExTorch.Native.aten_conv2d(inp3, w3, nil, [1, 1], [0, 0], [1, 1], 1)
    {us, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        ExTorch.Native.aten_conv2d(inp3, w3, nil, [1, 1], [0, 0], [1, 1], 1)
      end
    end)
    IO.puts("conv2d 256->64 k=1 s=1 p=0 input=56x56:   #{:io_lib.format(~c"~7.2f", [us / @iters / 1000])} ms/call")

    # MobileNetV2 depthwise: 96->96 k=3 groups=96 input=14x14
    inp4 = ExTorch.randn({1, 96, 14, 14})
    w4 = ExTorch.randn({96, 1, 3, 3})
    for _ <- 1..3, do: ExTorch.Native.aten_conv2d(inp4, w4, nil, [1, 1], [1, 1], [1, 1], 96)
    {us, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        ExTorch.Native.aten_conv2d(inp4, w4, nil, [1, 1], [1, 1], [1, 1], 96)
      end
    end)
    IO.puts("depthwise 96->96 k=3 groups=96 input=14x14: #{:io_lib.format(~c"~7.2f", [us / @iters / 1000])} ms/call")
  end
end

RawOp.run()
