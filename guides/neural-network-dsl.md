# Neural Network DSL

ExTorch provides an Elixir DSL for defining neural network architectures. Models can be created with random weights for experimentation, or loaded with pre-trained weights from TorchScript files.

## Defining a module

```elixir
defmodule MyClassifier do
  use ExTorch.NN.Module

  deflayer :conv1, ExTorch.NN.Conv2d, in_channels: 1, out_channels: 32, kernel_size: 3
  deflayer :relu1, ExTorch.NN.ReLU
  deflayer :pool, ExTorch.NN.MaxPool2d, kernel_size: 2
  deflayer :flatten, ExTorch.NN.Flatten
  deflayer :fc, ExTorch.NN.Linear, in_features: 32 * 13 * 13, out_features: 10

  def forward(model, x) do
    x
    |> layer(model, :conv1)
    |> layer(model, :relu1)
    |> layer(model, :pool)
    |> layer(model, :flatten)
    |> layer(model, :fc)
  end
end
```

`deflayer` declares a named layer with its type and options at compile time. `layer/3` applies a layer during the forward pass.

## Creating and using a model

```elixir
# Random weights
model = MyClassifier.new()
input = ExTorch.randn({1, 1, 28, 28})
output = MyClassifier.forward(model, input)
# => %ExTorch.Tensor{size: {1, 10}, ...}
```

## Inspecting parameters

```elixir
params = MyClassifier.parameters(model)
# => [
#   {"conv1.weight", #Tensor<[32, 1, 3, 3]>},
#   {"conv1.bias", #Tensor<[32]>},
#   {"fc.weight", #Tensor<[10, 5408]>},
#   {"fc.bias", #Tensor<[10]>}
# ]
```

## Loading pre-trained weights

There are two ways to use weights trained in Python:

### Option A: `from_jit` -- Use the JIT model directly

```elixir
model = MyClassifier.from_jit("trained_classifier.pt")
output = MyClassifier.predict(model, [input])
```

The JIT model's `forward` method handles all computation. The DSL definition is validated against the `.pt` file's submodules at load time -- if the architectures don't match, you get a clear error.

### Option B: `load_weights` -- Copy weights into DSL layers

```elixir
model = MyClassifier.load_weights("trained_classifier.pt")
output = MyClassifier.forward(model, input)
```

This creates the DSL layers, then copies matching parameter tensors from the `.pt` file. The result is a regular DSL model that runs through your Elixir `forward/2` function.

**When to use which:**

| | `from_jit` | `load_weights` |
|---|---|---|
| Runs | Python's forward logic | Your Elixir forward logic |
| Use when | You want exact Python behavior | Your forward differs (custom post-processing, different dropout) |
| Returns | `%JITBackedModel{}` | `%{layer => %Layer{}}` |

## Generating DSL from existing models

Don't have a DSL definition yet? ExTorch can introspect any `.pt` file and generate the Elixir source:

```elixir
model = ExTorch.JIT.load("resnet18.pt")
source = ExTorch.NN.Introspect.to_elixir(model, "ResNet18")
IO.puts(source)
```

Output:

```elixir
defmodule ResNet18 do
  use ExTorch.NN.Module

  deflayer :conv1, ExTorch.NN.Conv2d, in_channels: 3, out_channels: 64, kernel_size: 7
  deflayer :bn1, ExTorch.NN.BatchNorm2d
  deflayer :relu, ExTorch.NN.ReLU
  # ... (full architecture)

  def forward(x) do
    x
    |> layer(:conv1)
    |> layer(:bn1)
    |> layer(:relu)
    # ...
  end
end
```

You can paste this into your project and customize it.

## Available layers

### Convolutions
- `ExTorch.NN.Conv1d` -- `:in_channels`, `:out_channels`, `:kernel_size`, `:stride`, `:padding`, `:dilation`, `:groups`, `:bias`
- `ExTorch.NN.Conv2d` -- same options
- `ExTorch.NN.Conv3d` -- same options
- `ExTorch.NN.ConvTranspose1d` -- adds `:output_padding`
- `ExTorch.NN.ConvTranspose2d` -- adds `:output_padding`

### Pooling
- `ExTorch.NN.MaxPool1d` -- `:kernel_size`, `:stride`, `:padding`, `:dilation`, `:ceil_mode`
- `ExTorch.NN.MaxPool2d` -- same options
- `ExTorch.NN.AvgPool1d` -- `:kernel_size`, `:stride`, `:padding`, `:ceil_mode`, `:count_include_pad`
- `ExTorch.NN.AvgPool2d` -- same options
- `ExTorch.NN.AdaptiveAvgPool1d` -- `:output_size`
- `ExTorch.NN.AdaptiveAvgPool2d` -- `:output_h`, `:output_w`

### Normalization
- `ExTorch.NN.BatchNorm1d` -- `:num_features`, `:eps`, `:momentum`, `:affine`, `:track_running_stats`
- `ExTorch.NN.BatchNorm2d` -- same options
- `ExTorch.NN.LayerNorm` -- `:normalized_shape`, `:eps`, `:elementwise_affine`
- `ExTorch.NN.GroupNorm` -- `:num_groups`, `:num_channels`, `:eps`, `:affine`
- `ExTorch.NN.InstanceNorm1d` -- `:num_features`, `:eps`, `:momentum`, `:affine`, `:track_running_stats`
- `ExTorch.NN.InstanceNorm2d` -- same options

### Recurrent
- `ExTorch.NN.LSTM` -- `:input_size`, `:hidden_size`, `:num_layers`, `:bias`, `:batch_first`, `:dropout`, `:bidirectional`
- `ExTorch.NN.GRU` -- same options

### Attention
- `ExTorch.NN.MultiheadAttention` -- `:embed_dim`, `:num_heads`, `:dropout`, `:bias`

### Activations
`ReLU`, `LeakyReLU`, `GELU`, `ELU`, `SiLU` (Swish), `Mish`, `PReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`

### Other
- `ExTorch.NN.Linear` -- `:in_features`, `:out_features`, `:bias`
- `ExTorch.NN.Dropout` -- `:p`, `:inplace`
- `ExTorch.NN.Embedding` -- `:num_embeddings`, `:embedding_dim`, `:padding_idx`
- `ExTorch.NN.Flatten` -- `:start_dim`, `:end_dim`
- `ExTorch.NN.Unflatten` -- `:dim`, `:sizes`
