defmodule ExTorchTest.YolosTest do
  @moduledoc """
  End-to-end test for Hugging Face YOLOS — a transformer-based object
  detector — running through ExTorch.Export.

  YOLOS is a real modern detection architecture (ViT backbone + a fixed
  number of detection slots). Unlike anchor-based detectors (Faster R-CNN,
  Mask R-CNN, SSD, RetinaNet, FCOS) which all fail to export via
  torch.export — they hit GuardOnDataDependentSymNode inside torchvision's
  batched_nms — YOLOS exports cleanly because NMS is left to the user
  on the model output.

  Running YOLOS in ExTorch required two new op handlers
  (aten.slice.Tensor, aten.upsample_bicubic2d.vec). All other 17 ops in
  the graph were already supported.
  """
  use ExUnit.Case, async: false

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])
  @model_path Path.join(@fixtures_dir, "yolos.pt2")
  @input_shape {1, 3, 416, 416}
  @logits_shape {1, 100, 3}
  @boxes_shape {1, 100, 4}

  # YOLOS has many fused ops (LayerNorm, scaled_dot_product_attention,
  # gelu) that accumulate slightly differently in the dispatcher path
  # vs PyTorch's eager path. Loosen tolerance accordingly — we're
  # verifying functional equivalence, not bit-identical math.
  @tol_rtol 1.0e-3
  @tol_atol 1.0e-4

  setup_all do
    unless File.exists?(@model_path),
      do: flunk("Run: .venv/bin/python test/fixtures/generate_yolos.py")

    :ok
  end

  defp read_bin(name, shape) do
    path = Path.join(@fixtures_dir, "#{name}.bin")
    ExTorch.Native.from_binary(File.read!(path), shape, :float32)
  end

  defp assert_yolos_path(forward_fn) do
    model = ExTorch.Export.load(@model_path)
    input = read_bin("yolos_input", @input_shape)

    [logits, boxes] = forward_fn.(model, input)

    assert %ExTorch.Tensor{} = logits
    assert %ExTorch.Tensor{} = boxes
    assert logits.size == @logits_shape
    assert boxes.size == @boxes_shape

    expected_logits = read_bin("yolos_logits", @logits_shape)
    expected_boxes = read_bin("yolos_boxes", @boxes_shape)

    assert ExTorch.allclose(logits, expected_logits, @tol_rtol, @tol_atol),
           "YOLOS logits diverge from PyTorch reference"

    assert ExTorch.allclose(boxes, expected_boxes, @tol_rtol, @tol_atol),
           "YOLOS pred_boxes diverge from PyTorch reference"
  end

  test "forward/2 (interpreter) matches PyTorch" do
    assert_yolos_path(fn m, x -> ExTorch.Export.forward(m, [x]) end)
  end

  test "forward_native/2 matches PyTorch" do
    assert_yolos_path(fn m, x -> ExTorch.Export.forward_native(m, [x]) end)
  end

  test "forward_compiled/2 matches PyTorch" do
    assert_yolos_path(fn m, x -> ExTorch.Export.forward_compiled(m, [x]) end)
  end
end
