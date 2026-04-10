# GPU version of bench/raw_op.exs: times raw aten_conv2d NIF calls with
# CUDA tensors to measure per-op dispatch and kernel launch overhead.

unless ExTorch.Native.cuda_is_available() do
  IO.puts("ERROR: CUDA not available")
  System.halt(1)
end

ExTorch.set_grad_enabled(false)

defmodule RawOpGpu do
  @iters 200

  defp cuda_tensor(shape), do: ExTorch.Tensor.to(ExTorch.randn(shape), device: :cuda)

  def bench(name, input, weight, stride, padding, dilation, groups) do
    # Warmup + sync so the first timed call is steady-state.
    for _ <- 1..5 do
      ExTorch.Native.aten_conv2d(input, weight, nil, stride, padding, dilation, groups)
    end
    ExTorch.cuda_synchronize()

    {us, _} = :timer.tc(fn ->
      for _ <- 1..@iters do
        ExTorch.Native.aten_conv2d(input, weight, nil, stride, padding, dilation, groups)
      end
      # Drain the stream before stopping the clock so we measure the
      # actual kernel time, not just queue latency.
      ExTorch.cuda_synchronize()
    end)
    IO.puts("#{String.pad_trailing(name, 48)} #{:io_lib.format(~c"~7.3f", [us / @iters / 1000])} ms/call")
  end

  def run do
    # ResNet50 first conv
    bench("conv2d 3->64 k=7 s=2 p=3 input=224x224",
      cuda_tensor({1, 3, 224, 224}), cuda_tensor({64, 3, 7, 7}),
      [2, 2], [3, 3], [1, 1], 1)

    # ResNet layer1 conv
    bench("conv2d 64->64 k=3 s=1 p=1 input=56x56",
      cuda_tensor({1, 64, 56, 56}), cuda_tensor({64, 64, 3, 3}),
      [1, 1], [1, 1], [1, 1], 1)

    # ResNet50 bottleneck 1x1
    bench("conv2d 256->64 k=1 s=1 p=0 input=56x56",
      cuda_tensor({1, 256, 56, 56}), cuda_tensor({64, 256, 1, 1}),
      [1, 1], [0, 0], [1, 1], 1)

    # MobileNetV2 depthwise
    bench("depthwise 96->96 k=3 groups=96 input=14x14",
      cuda_tensor({1, 96, 14, 14}), cuda_tensor({96, 1, 3, 3}),
      [1, 1], [1, 1], [1, 1], 96)

    # Deeper bottleneck
    bench("conv2d 512->512 k=3 s=1 p=1 input=7x7",
      cuda_tensor({1, 512, 7, 7}), cuda_tensor({512, 512, 3, 3}),
      [1, 1], [1, 1], [1, 1], 1)
  end
end

RawOpGpu.run()
