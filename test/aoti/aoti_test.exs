defmodule ExTorchTest.AOTITest do
  use ExUnit.Case, async: true

  @fixtures_dir Path.join([__DIR__, "..", "fixtures"])

  setup_all do
    pt2_path = Path.join(@fixtures_dir, "simple_mlp.pt2")

    unless File.exists?(pt2_path) do
      {_, 0} = System.cmd(
        Path.join([File.cwd!(), ".venv", "bin", "python"]),
        ["-c", """
        import torch, torch.nn as nn
        from torch._inductor import aoti_compile_and_package
        torch.manual_seed(42)
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(20, 5)
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))
        m = M()
        m.eval()
        e = torch.export.export(m, (torch.randn(1, 10),))
        aoti_compile_and_package(e, package_path='#{pt2_path}')
        """])
    end

    :ok
  end

  describe "available?/0" do
    test "returns boolean" do
      assert is_boolean(ExTorch.AOTI.available?())
    end
  end

  describe "load/2" do
    test "loads a .pt2 model" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      model = ExTorch.AOTI.load(path)
      assert %ExTorch.AOTI.Model{} = model
    end
  end

  describe "forward/2" do
    test "runs inference" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      model = ExTorch.AOTI.load(path)

      input = ExTorch.randn({1, 10})
      outputs = ExTorch.AOTI.forward(model, [input])

      assert is_list(outputs)
      assert length(outputs) == 1
      assert %ExTorch.Tensor{} = hd(outputs)
      assert hd(outputs).size == {1, 5}
    end

    test "handles batch inference" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      model = ExTorch.AOTI.load(path)

      input = ExTorch.randn({8, 10})
      [output] = ExTorch.AOTI.forward(model, [input])
      assert output.size == {8, 5}
    end
  end

  describe "metadata/1" do
    test "returns metadata map" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      model = ExTorch.AOTI.load(path)

      meta = ExTorch.AOTI.metadata(model)
      assert is_map(meta)
      assert Map.has_key?(meta, "AOTI_DEVICE_KEY")
    end
  end

  describe "constant_names/1" do
    test "returns parameter names" do
      path = Path.join(@fixtures_dir, "simple_mlp.pt2")
      model = ExTorch.AOTI.load(path)

      names = ExTorch.AOTI.constant_names(model)
      assert is_list(names)
      assert "fc1.weight" in names
      assert "fc1.bias" in names or "fc2.weight" in names
    end
  end
end
