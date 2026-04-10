# Probe the MKLDNN weight pre-packing path:
#   1. Is the pre-packed conv output numerically identical to at::conv2d?
#   2. How much faster is it per call?

ExTorch.set_grad_enabled(false)
ExTorch.clear_cpu_affinity()

defmodule PrepackProbe do
  @iters 100

  def test(name, in_shape, k_shape, stride, padding, dilation, groups) do
    input = ExTorch.randn(in_shape)
    weight = ExTorch.randn(k_shape)

    # Correctness: compare regular at::conv2d to mkldnn_convolution with packed weight
    out_regular = ExTorch.Native.aten_conv2d(input, weight, nil, stride, padding, dilation, groups)
    packed = ExTorch.Native.aten_mkldnn_reorder_conv2d_weight(weight, padding, stride, dilation, groups)
    out_packed = ExTorch.Native.aten_mkldnn_convolution(input, packed, nil, padding, stride, dilation, groups)

    match = ExTorch.allclose(out_regular, out_packed, 1.0e-4, 1.0e-5)
    diff = ExTorch.sub(out_regular, out_packed) |> ExTorch.mul(ExTorch.sub(out_regular, out_packed))
    max_sq = ExTorch.max(diff) |> ExTorch.Tensor.item()

    # Warmup
    for _ <- 1..5, do: ExTorch.Native.aten_conv2d(input, weight, nil, stride, padding, dilation, groups)
    for _ <- 1..5, do: ExTorch.Native.aten_mkldnn_convolution(input, packed, nil, padding, stride, dilation, groups)

    # Time regular
    {us_reg, _} = :timer.tc(fn ->
      for _ <- 1..@iters, do: ExTorch.Native.aten_conv2d(input, weight, nil, stride, padding, dilation, groups)
    end)
    # Time pre-packed
    {us_pack, _} = :timer.tc(fn ->
      for _ <- 1..@iters, do: ExTorch.Native.aten_mkldnn_convolution(input, packed, nil, padding, stride, dilation, groups)
    end)

    reg_ms = us_reg / @iters / 1000
    pack_ms = us_pack / @iters / 1000
    speedup = reg_ms / pack_ms

    IO.puts("#{String.pad_trailing(name, 42)}  regular=#{:io_lib.format(~c"~7.2f", [reg_ms])} ms  packed=#{:io_lib.format(~c"~7.2f", [pack_ms])} ms  speedup=#{:io_lib.format(~c"~5.2f", [speedup])}x  match=#{match}  max_diff=#{:io_lib.format(~c"~.2e", [:math.sqrt(max_sq)])}")
  end
end

# ResNet50 first conv
PrepackProbe.test("conv 3->64 k=7 s=2 p=3 224x224",
  {1, 3, 224, 224}, {64, 3, 7, 7}, [2, 2], [3, 3], [1, 1], 1)

# ResNet layer1 conv
PrepackProbe.test("conv 64->64 k=3 s=1 p=1 56x56",
  {1, 64, 56, 56}, {64, 64, 3, 3}, [1, 1], [1, 1], [1, 1], 1)

# ResNet50 bottleneck 1x1
PrepackProbe.test("conv 256->64 k=1 56x56",
  {1, 256, 56, 56}, {64, 256, 1, 1}, [1, 1], [0, 0], [1, 1], 1)

# MobileNetV2 depthwise
PrepackProbe.test("depthwise 96->96 k=3 g=96 14x14",
  {1, 96, 14, 14}, {96, 1, 3, 3}, [1, 1], [1, 1], [1, 1], 96)

# Deeper bottleneck conv
PrepackProbe.test("conv 512->512 k=3 7x7",
  {1, 512, 7, 7}, {512, 512, 3, 3}, [1, 1], [1, 1], [1, 1], 1)
