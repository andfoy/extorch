# Real-World Model Deployment Benchmark
#
# Runs exported models through the full ExTorch inference stack and
# reports latency across all three paths (interpreter, native, AOTI).
#
# Usage:
#   .venv/bin/python examples/models/export_models.py [--device cuda] [--aoti]
#   mix run examples/models/benchmark.exs

defmodule ModelBenchmark do
  @fixtures Path.join([__DIR__, "fixtures"])
  @warmup 5
  @iters 30

  @models [
    # {name, input_shapes, tags}
    {"clip_visual",           [{1, 3, 224, 224}],                [:production, :multimodal]},
    {"distilbert",            [{1, 128}, {1, 128}],              [:production, :nlp]},
    {"mobilenet_v3_small",    [{1, 3, 224, 224}],                [:production, :classification]},
    {"efficientnet_b0",       [{1, 3, 224, 224}],                [:production, :classification]},
    {"resnet50_prod",         [{1, 3, 224, 224}],                [:production, :classification]},
    {"vit_b_16_prod",         [{1, 3, 224, 224}],                [:production, :transformer]},
    {"deeplabv3_mobilenet",   [{1, 3, 320, 320}],                [:production, :segmentation]},
    {"whisper_tiny_encoder",  [{1, 80, 3000}],                   [:production, :speech]},
  ]

  def run do
    ExTorch.set_grad_enabled(false)

    device = if ExTorch.Native.cuda_is_available(), do: :cuda, else: :cpu
    IO.puts("Device: #{device}")
    IO.puts("Models directory: #{@fixtures}\n")

    IO.puts(format_header())
    IO.puts(String.duplicate("-", 80))

    for {name, input_shapes, tags} <- @models do
      run_model(name, input_shapes, tags, device)
    end
  end

  defp run_model(name, input_shapes, tags, device) do
    pt2_path = Path.join(@fixtures, "#{name}.pt2")
    aoti_path = Path.join(@fixtures, "#{name}_aoti.pt2")

    if not File.exists?(pt2_path) do
      IO.puts(format_row(name, tags, "SKIP", "SKIP", "SKIP"))
    else
    inputs = load_inputs(name, input_shapes, device)

    # Export interpreter
    interp_ms = try do
      model = ExTorch.Export.load(pt2_path, device: device)
      bench(fn -> ExTorch.Export.forward(model, inputs) end, device)
    rescue
      e -> "ERR:#{String.slice(Exception.message(e), 0..15)}"
    end

    # Export native (C++ graph executor)
    native_ms = try do
      model = ExTorch.Export.load(pt2_path, device: device)
      bench(fn -> ExTorch.Export.forward_native(model, inputs) end, device)
    rescue
      e -> "ERR:#{String.slice(Exception.message(e), 0..15)}"
    end

    # AOTI
    aoti_ms = if File.exists?(aoti_path) and ExTorch.AOTI.available?() do
      try do
        aoti_model = ExTorch.AOTI.load(aoti_path,
          device_index: if(device == :cuda, do: 0, else: -1))
        bench(fn -> ExTorch.AOTI.forward(aoti_model, inputs) end, device)
      rescue
        e -> "ERR:#{String.slice(Exception.message(e), 0..15)}"
      end
    else
      "N/A"
    end

    IO.puts(format_row(name, tags, interp_ms, native_ms, aoti_ms))
    end # if File.exists?
  end

  defp bench(fun, device) do
    # Warmup
    for _ <- 1..@warmup, do: fun.()
    if device == :cuda, do: ExTorch.cuda_synchronize()

    samples = for _ <- 1..@iters do
      {us, _} = :timer.tc(fn ->
        fun.()
        if device == :cuda, do: ExTorch.cuda_synchronize()
      end)
      us / 1000.0
    end

    sorted = Enum.sort(samples)
    median = Enum.at(sorted, div(@iters, 2))
    :erlang.float_to_binary(median, decimals: 2)
  end

  defp load_inputs(name, input_shapes, device) do
    Enum.with_index(input_shapes)
    |> Enum.map(fn {shape, idx} ->
      suffix = if length(input_shapes) > 1, do: "_input_#{idx}", else: "_input"
      bin_path = Path.join(@fixtures, "#{name}#{suffix}.bin")
      dtype = if elem(shape, 1) > 200, do: :float32, else: :int64

      # Detect dtype from shape heuristic:
      # image inputs have channels (3) in position 1 or values > 100 in dims
      dtype = cond do
        tuple_size(shape) == 4 -> :float32  # image: {B, C, H, W}
        tuple_size(shape) == 3 -> :float32  # audio: {B, mels, frames}
        true -> :int64                       # token ids: {B, seq_len}
      end

      tensor = if File.exists?(bin_path) do
        ExTorch.Native.from_binary(File.read!(bin_path), shape, dtype)
      else
        # Generate random input
        case dtype do
          :float32 -> ExTorch.randn(shape)
          :int64 -> ExTorch.randint(0, 30000, shape, dtype: :long)
        end
      end

      if device == :cuda, do: ExTorch.Tensor.to(tensor, device: :cuda), else: tensor
    end)
  end

  defp format_header do
    :io_lib.format("~-24s ~-14s ~10s ~10s ~10s",
      [~c"model", ~c"type", ~c"interp", ~c"native", ~c"aoti"])
    |> to_string()
  end

  defp format_row(name, tags, interp, native, aoti) do
    tag_str = tags |> Enum.at(1, :unknown) |> to_string()
    format_val = fn
      v when is_binary(v) -> v
      v -> v
    end
    :io_lib.format("~-24s ~-14s ~10s ~10s ~10s",
      [String.to_charlist(name),
       String.to_charlist(tag_str),
       String.to_charlist(format_val.(interp)),
       String.to_charlist(format_val.(native)),
       String.to_charlist(format_val.(aoti))])
    |> to_string()
  end
end

ModelBenchmark.run()
