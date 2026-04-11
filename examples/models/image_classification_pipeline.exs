# End-to-End Image Classification Pipeline
#
# Full production pipeline: load image → preprocess → infer → decode labels
# Demonstrates how Python's torchvision.transforms maps to Elixir.
#
# Usage:
#   .venv/bin/python examples/models/export_models.py
#   mix run examples/models/image_classification_pipeline.exs [image_path]
#
# If no image is provided, generates a random input.

defmodule ImageClassificationPipeline do
  @fixtures Path.join([__DIR__, "fixtures"])

  # ImageNet normalization stats
  @imagenet_mean [0.485, 0.456, 0.406]
  @imagenet_std [0.229, 0.224, 0.225]

  # Top-5 ImageNet class names (subset for demo)
  @imagenet_labels %{
    0 => "tench", 1 => "goldfish", 2 => "great white shark",
    386 => "African elephant", 388 => "giant panda",
    948 => "Granny Smith (apple)", 954 => "banana",
    985 => "daisy", 309 => "bee", 207 => "golden retriever"
  }

  def run(image_path \\ nil) do
    ExTorch.set_grad_enabled(false)

    device = if ExTorch.Native.cuda_is_available(), do: :cuda, else: :cpu
    model_name = "resnet50_prod"
    pt2_path = Path.join(@fixtures, "#{model_name}.pt2")

    unless File.exists?(pt2_path) do
      IO.puts("Missing model. Run: .venv/bin/python examples/models/export_models.py")
      System.halt(1)
    end

    IO.puts("=== Image Classification Pipeline ===")
    IO.puts("Model: #{model_name}")
    IO.puts("Device: #{device}\n")

    # =========================================================================
    # Step 1: Load Model
    # =========================================================================
    IO.puts("1. Loading model...")
    {us, model} = :timer.tc(fn -> ExTorch.Export.load(pt2_path, device: device) end)
    IO.puts("   Loaded in #{Float.round(us / 1000, 1)}ms " <>
            "(#{length(model.schema.graph)} graph nodes)\n")

    # =========================================================================
    # Step 2: Preprocess Image
    #
    # Python equivalent:
    #   transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    #   ])
    # =========================================================================
    IO.puts("2. Preprocessing...")

    input_tensor = if image_path && File.exists?(image_path) do
      preprocess_image(image_path)
    else
      IO.puts("   (no image provided, using random input)")
      ExTorch.randn({1, 3, 224, 224})
    end

    input_tensor = if device == :cuda,
      do: ExTorch.Tensor.to(input_tensor, device: :cuda),
      else: input_tensor

    IO.puts("   Input shape: #{inspect(input_tensor.size)}")
    IO.puts("   Input device: #{inspect(input_tensor.device)}\n")

    # =========================================================================
    # Step 3: Run Inference (all three paths)
    # =========================================================================
    IO.puts("3. Running inference...\n")

    # Native path (recommended for production)
    {us_native, output} = :timer.tc(fn ->
      ExTorch.Export.forward_native(model, [input_tensor])
    end)
    if device == :cuda, do: ExTorch.cuda_synchronize()
    IO.puts("   Native:      #{Float.round(us_native / 1000, 2)}ms")

    # Interpreter path (for debugging)
    {us_interp, _} = :timer.tc(fn ->
      ExTorch.Export.forward(model, [input_tensor])
    end)
    if device == :cuda, do: ExTorch.cuda_synchronize()
    IO.puts("   Interpreter: #{Float.round(us_interp / 1000, 2)}ms\n")

    # =========================================================================
    # Step 4: Postprocess — Softmax + Top-5
    # =========================================================================
    IO.puts("4. Postprocessing...")

    # Move to CPU for postprocessing
    output_cpu = if device == :cuda,
      do: ExTorch.Tensor.to(output, device: :cpu),
      else: output

    # Softmax
    probs = ExTorch.functional_softmax(output_cpu, 1)

    # Top-5
    {top_values, top_indices} = ExTorch.topk(probs, 5)

    IO.puts("\n   Top-5 predictions:")
    for i <- 0..4 do
      idx = ExTorch.select(top_indices, 1, i) |> ExTorch.Tensor.item() |> round()
      prob = ExTorch.select(top_values, 1, i) |> ExTorch.Tensor.item()
      label = Map.get(@imagenet_labels, idx, "class_#{idx}")
      IO.puts("   #{i + 1}. #{label} (#{Float.round(prob * 100, 2)}%)")
    end
  end

  # Preprocessing pipeline using Nx (or ExTorch directly)
  #
  # For production, you'd use the Image or NxImage library:
  #   Image.open(path) |> Image.resize(256) |> Image.center_crop(224)
  #
  # Here we show the ExTorch-native approach.
  defp preprocess_image(path) do
    # Read raw bytes and decode via ExTorch.Vision if available
    data = File.read!(path)
    data_tensor = ExTorch.Native.from_binary(data, {byte_size(data)}, :uint8)

    # Try decoding with ExTorch.Vision (if available)
    image = try do
      ExTorch.Vision.setup!()
      decoded = ExTorch.Vision.decode_image(data_tensor, 2)  # mode=2 for RGB
      IO.puts("   Decoded with ExTorch.Vision: #{inspect(decoded.size)}")
      decoded
    rescue
      _ ->
        IO.puts("   ExTorch.Vision not available, using random input")
        # Fallback: generate random 224x224 image
        ExTorch.randint(0, 255, {3, 224, 224}, dtype: :uint8)
    end

    # Convert to float and normalize
    # Python: transforms.ToTensor() does /255 and HWC->CHW
    float_image = ExTorch.Tensor.to(image, dtype: :float32)
    float_image = ExTorch.tensor_div(float_image, 255.0)

    # Normalize with ImageNet stats
    # Python: transforms.Normalize(mean, std)
    mean = ExTorch.tensor(@imagenet_mean) |> ExTorch.view({3, 1, 1})
    std = ExTorch.tensor(@imagenet_std) |> ExTorch.view({3, 1, 1})
    normalized = ExTorch.tensor_div(ExTorch.sub(float_image, mean), std)

    # Add batch dimension
    ExTorch.unsqueeze(normalized, 0)
  end
end

# Run with optional image path argument
image_path = case System.argv() do
  [path | _] -> path
  _ -> nil
end

ImageClassificationPipeline.run(image_path)
