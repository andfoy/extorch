defmodule ExTorchTest.NN.LayersTest do
  use ExUnit.Case, async: true

  describe "linear" do
    test "creates a linear layer with correct shapes" do
      layer = ExTorch.NN.linear(10, 5)
      assert %ExTorch.NN.Layer{type_name: "Linear"} = layer

      params = ExTorch.NN.parameters(layer)
      assert length(params) == 2

      {_name, weight} = Enum.find(params, fn {n, _} -> n == "weight" end)
      assert weight.size == {5, 10}

      {_name, bias} = Enum.find(params, fn {n, _} -> n == "bias" end)
      assert bias.size == {5}
    end

    test "creates a linear layer without bias" do
      layer = ExTorch.NN.linear(10, 5, bias: false)
      params = ExTorch.NN.parameters(layer)
      assert length(params) == 1
    end

    test "forward pass produces correct output shape" do
      layer = ExTorch.NN.linear(10, 5)
      input = ExTorch.randn({3, 10})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {3, 5}
    end
  end

  describe "conv2d" do
    test "creates a conv2d layer" do
      layer = ExTorch.NN.conv2d(3, 16, 3, padding: 1)
      assert %ExTorch.NN.Layer{type_name: "Conv2d"} = layer
    end

    test "forward pass produces correct output shape" do
      layer = ExTorch.NN.conv2d(3, 16, 3, padding: 1)
      input = ExTorch.randn({1, 3, 8, 8})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {1, 16, 8, 8}
    end
  end

  describe "activations" do
    test "relu" do
      layer = ExTorch.NN.relu()
      assert %ExTorch.NN.Layer{type_name: "ReLU"} = layer

      input = ExTorch.tensor([-1.0, 0.0, 1.0, 2.0])
      output = ExTorch.NN.forward(input, layer)
      expected = ExTorch.tensor([0.0, 0.0, 1.0, 2.0])
      assert ExTorch.allclose(output, expected)
    end

    test "gelu" do
      layer = ExTorch.NN.gelu()
      assert %ExTorch.NN.Layer{type_name: "GELU"} = layer
    end

    test "sigmoid" do
      layer = ExTorch.NN.sigmoid()
      input = ExTorch.tensor([0.0])
      output = ExTorch.NN.forward(input, layer)
      expected = ExTorch.tensor([0.5])
      assert ExTorch.allclose(output, expected)
    end
  end

  describe "batch_norm1d" do
    test "creates and runs forward" do
      layer = ExTorch.NN.batch_norm1d(10)
      assert %ExTorch.NN.Layer{type_name: "BatchNorm1d"} = layer

      input = ExTorch.randn({4, 10})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {4, 10}
    end
  end

  describe "dropout" do
    test "creates dropout layer" do
      layer = ExTorch.NN.dropout(p: 0.5)
      assert %ExTorch.NN.Layer{type_name: "Dropout"} = layer
    end
  end

  describe "embedding" do
    test "creates and runs forward" do
      layer = ExTorch.NN.embedding(100, 32)
      assert %ExTorch.NN.Layer{type_name: "Embedding"} = layer

      input = ExTorch.tensor([0, 5, 10], dtype: :int64)
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {3, 32}
    end
  end

  describe "conv3d" do
    test "creates and runs forward" do
      layer = ExTorch.NN.conv3d(1, 8, 3)
      assert %ExTorch.NN.Layer{type_name: "Conv3d"} = layer

      input = ExTorch.randn({1, 1, 8, 8, 8})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {1, 8, 6, 6, 6}
    end
  end

  describe "conv_transpose2d" do
    test "creates and runs forward" do
      layer = ExTorch.NN.conv_transpose2d(16, 3, 3, stride: 2, padding: 1, output_padding: 1)
      assert %ExTorch.NN.Layer{type_name: "ConvTranspose2d"} = layer

      input = ExTorch.randn({1, 16, 4, 4})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {1, 3, 8, 8}
    end
  end

  describe "pooling" do
    test "max_pool2d halves spatial dims" do
      layer = ExTorch.NN.max_pool2d(2)
      input = ExTorch.randn({1, 1, 8, 8})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {1, 1, 4, 4}
    end

    test "avg_pool2d" do
      layer = ExTorch.NN.avg_pool2d(2)
      input = ExTorch.randn({1, 1, 6, 6})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {1, 1, 3, 3}
    end

    test "adaptive_avg_pool2d reduces to target size" do
      layer = ExTorch.NN.adaptive_avg_pool2d(1, 1)
      input = ExTorch.randn({2, 16, 7, 7})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {2, 16, 1, 1}
    end

    test "max_pool1d" do
      layer = ExTorch.NN.max_pool1d(3, stride: 2)
      input = ExTorch.randn({1, 4, 10})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {1, 4, 4}
    end
  end

  describe "normalization" do
    test "group_norm" do
      layer = ExTorch.NN.group_norm(4, 16)
      input = ExTorch.randn({2, 16, 4, 4})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {2, 16, 4, 4}
    end

    test "instance_norm2d" do
      layer = ExTorch.NN.instance_norm2d(8)
      input = ExTorch.randn({2, 8, 4, 4})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {2, 8, 4, 4}
    end

    test "layer_norm" do
      layer = ExTorch.NN.layer_norm([10])
      input = ExTorch.randn({3, 10})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {3, 10}
    end
  end

  describe "additional activations" do
    test "leaky_relu" do
      layer = ExTorch.NN.leaky_relu(negative_slope: 0.2)
      input = ExTorch.tensor([-1.0, 0.0, 1.0])
      output = ExTorch.NN.forward(input, layer)
      expected = ExTorch.tensor([-0.2, 0.0, 1.0])
      assert ExTorch.allclose(output, expected)
    end

    test "elu" do
      layer = ExTorch.NN.elu()
      assert %ExTorch.NN.Layer{type_name: "ELU"} = layer
    end

    test "silu" do
      layer = ExTorch.NN.silu()
      assert %ExTorch.NN.Layer{type_name: "SiLU"} = layer
      # SiLU(0) = 0 * sigmoid(0) = 0
      output = ExTorch.NN.forward(ExTorch.tensor([0.0]), layer)
      expected = ExTorch.tensor([0.0])
      assert ExTorch.allclose(output, expected)
    end

    test "mish" do
      layer = ExTorch.NN.mish()
      assert %ExTorch.NN.Layer{type_name: "Mish"} = layer
    end

    test "prelu" do
      layer = ExTorch.NN.prelu()
      assert %ExTorch.NN.Layer{type_name: "PReLU"} = layer
      params = ExTorch.NN.parameters(layer)
      assert length(params) == 1
    end

    test "log_softmax" do
      layer = ExTorch.NN.log_softmax(0)
      assert %ExTorch.NN.Layer{type_name: "LogSoftmax"} = layer
    end
  end

  describe "utility layers" do
    test "flatten" do
      layer = ExTorch.NN.flatten()
      input = ExTorch.randn({2, 3, 4, 5})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {2, 60}
    end

    test "flatten with custom dims" do
      layer = ExTorch.NN.flatten(start_dim: 0)
      input = ExTorch.randn({2, 3, 4})
      output = ExTorch.NN.forward(input, layer)
      assert output.size == {24}
    end
  end

  describe "eval/train" do
    test "sets mode without error" do
      layer = ExTorch.NN.linear(10, 5)
      assert :ok = ExTorch.NN.eval(layer)
      assert :ok = ExTorch.NN.train(layer)
    end
  end

  describe "copy_parameters" do
    test "copies weights between layers" do
      src = ExTorch.NN.linear(10, 5)
      dst = ExTorch.NN.linear(10, 5)

      src_params = ExTorch.NN.parameters(src)
      ExTorch.NN.copy_parameters(dst, src_params)
      dst_params = ExTorch.NN.parameters(dst)

      Enum.zip(src_params, dst_params)
      |> Enum.each(fn {{name, src_t}, {name2, dst_t}} ->
        assert name == name2
        assert ExTorch.equal(src_t, dst_t)
      end)
    end
  end
end
