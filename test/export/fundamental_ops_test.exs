defmodule ExTorchTest.FundamentalOpsTest do
  @moduledoc """
  Regression test for the fundamental Export op handlers:
  matmul, stack, concat, arange, ones, to.dtype.

  Fixture generator: test/fixtures/generate_fundamental_ops.py
  """
  use ExUnit.Case, async: false

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])
  @model_path Path.join(@fixtures_dir, "fundamental_ops.pt2")

  setup_all do
    unless File.exists?(@model_path),
      do: flunk("Run: .venv/bin/python test/fixtures/generate_fundamental_ops.py")

    :ok
  end

  defp read_bin(name, shape) do
    path = Path.join(@fixtures_dir, "#{name}.bin")
    ExTorch.Native.from_binary(File.read!(path), shape, :float32)
  end

  test "matmul, stack, concat, arange, ones, to.dtype all execute and match PyTorch" do
    model = ExTorch.Export.load(@model_path)
    input = read_bin("fundamental_ops_input", {4, 3})
    expected = read_bin("fundamental_ops_output", {1})

    for {label, runner} <- [
          {"forward/2", fn -> ExTorch.Export.forward(model, [input]) end},
          {"forward_native/2", fn -> ExTorch.Export.forward_native(model, [input]) end},
          {"forward_compiled/2",
           fn ->
             m = ExTorch.Export.load(@model_path)
             ExTorch.Export.forward_compiled(m, [input])
           end}
        ] do
      out = runner.()

      assert %ExTorch.Tensor{} = out, "#{label}: non-tensor output"

      # Expected is saved as shape (1,); model emits 0-d.
      # Reshape both to (1,) for a direct element-wise compare.
      got = ExTorch.reshape(out, {1})

      assert ExTorch.allclose(got, expected, 1.0e-3, 1.0e-4),
             "#{label}: output diverges from PyTorch reference"
    end
  end
end
