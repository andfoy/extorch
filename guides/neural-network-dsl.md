# Neural Network DSL

ExTorch provides an Elixir DSL for defining neural network architectures.
Models can be created with random weights for experimentation, loaded with
pre-trained weights from TorchScript or `torch.export.save` archives, or
composed from nested sub-modules.

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

`deflayer` declares a named layer with its type and options at compile time.
`layer/3` applies a layer during the forward pass.

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

There are three ways to use weights trained in Python:

### Option A: `load_weights_from_export` (recommended -- JIT-free)

Load weights from a `torch.export.save` `.pt2` archive. No `torch.jit`, no
Python at runtime:

```python
# Python
exported = torch.export.export(model, (example_input,))
torch.export.save(exported, "trained.pt2")
```

```elixir
model = MyClassifier.load_weights_from_export("trained.pt2")
output = MyClassifier.forward(model, input)
```

Creates the DSL layers, then copies matching parameter tensors from the `.pt2`
archive. The result is a regular DSL model that runs through your Elixir
`forward/2` function.

### Option B: `load_weights` (from TorchScript)

```elixir
model = MyClassifier.load_weights("trained.pt")
output = MyClassifier.forward(model, input)
```

Same as Option A but reads weights from a TorchScript `.pt` file. Uses the
deprecated `torch.jit` path.

### Option C: `from_jit` (delegate to JIT forward)

```elixir
model = MyClassifier.from_jit("trained.pt")
output = MyClassifier.predict(model, [input])
```

The JIT model's `forward` method handles all computation. The DSL definition
is validated against the `.pt` file's submodules at load time.

**When to use which:**

| | `load_weights_from_export` | `load_weights` | `from_jit` |
|---|---|---|---|
| Source format | `.pt2` (torch.export) | `.pt` (TorchScript) | `.pt` (TorchScript) |
| Runs | Your Elixir forward | Your Elixir forward | Python's forward |
| JIT dependency | No | Yes (deprecated) | Yes (deprecated) |
| Returns | `%{layer => %Layer{}}` | `%{layer => %Layer{}}` | `%JITBackedModel{}` |

## Composing modules

DSL modules can be nested inside other DSL modules:

```elixir
defmodule FeatureBlock do
  use ExTorch.NN.Module

  deflayer :conv, ExTorch.NN.Conv2d, in_channels: 16, out_channels: 16, kernel_size: 3, padding: 1
  deflayer :bn, ExTorch.NN.BatchNorm2d, num_features: 16
  deflayer :relu, ExTorch.NN.ReLU

  def forward(model, x) do
    x |> layer(model, :conv) |> layer(model, :bn) |> layer(model, :relu)
  end
end

defmodule SmallNet do
  use ExTorch.NN.Module

  deflayer :stem, ExTorch.NN.Conv2d, in_channels: 1, out_channels: 16, kernel_size: 3, padding: 1
  deflayer :block1, FeatureBlock
  deflayer :block2, FeatureBlock
  deflayer :pool, ExTorch.NN.AdaptiveAvgPool2d, output_h: 1, output_w: 1
  deflayer :flatten, ExTorch.NN.Flatten
  deflayer :fc, ExTorch.NN.Linear, in_features: 16, out_features: 10

  def forward(model, x) do
    x
    |> layer(model, :stem)
    |> layer(model, :block1)
    |> layer(model, :block2)
    |> layer(model, :pool)
    |> layer(model, :flatten)
    |> layer(model, :fc)
  end
end
```

When `layer/3` encounters a nested DSL module, it calls the sub-module's
`forward/2` automatically. Parameters are namespaced with dotted paths
(`"block1.conv.weight"`, `"block1.bn.bias"`, etc.).

## Generating DSL from existing models

Don't have a DSL definition yet? ExTorch can generate one from either a
`torch.export.save` archive or a TorchScript model.

### From torch.export (recommended)

```elixir
source = ExTorch.Export.to_elixir("model.pt2", "MyModel")
IO.puts(source)
```

This parses the ATen graph IR from the `.pt2` archive, maps operations to
ExTorch NN layer types, and infers layer parameters from weight tensor shapes.
It works without Python or JIT.

The generator performs **data flow analysis** on the computation graph to
correctly handle branching architectures like ResNets. Skip connections,
downsample paths, and merge points are expressed with explicit variable
assignments:

```elixir
# Generated from a ResNet -- note the residual connections
def forward(model, x) do
  conv2d = x |> layer(model, :conv1)
  batch_norm = conv2d |> layer(model, :bn1)
  relu_ = batch_norm |> layer(model, :relu_)
  max_pool2d = relu_ |> layer(model, :max_pool2d)
  conv2d_1 = max_pool2d |> layer(model, :layer1_0_conv1)
  batch_norm_1 = conv2d_1 |> layer(model, :layer1_0_bn1)
  relu__1 = batch_norm_1 |> layer(model, :relu__1)
  conv2d_2 = relu__1 |> layer(model, :layer1_0_conv2)
  batch_norm_2 = conv2d_2 |> layer(model, :layer1_0_bn2)
  add_ = ExTorch.add(batch_norm_2, max_pool2d)    # skip connection
  relu__2 = add_ |> layer(model, :relu__2)
  # ...
end
```

The generator tracks which values are consumed by multiple downstream nodes
(branch points) and which nodes take inputs from different branches (merge
points), emitting proper variable bindings for each.

### From TorchScript

```elixir
model = ExTorch.JIT.load("model.pt")
source = ExTorch.NN.Introspect.to_elixir(model, "MyModel")
IO.puts(source)
```

The JIT introspection recursively expands Sequential containers into their
leaf layers. The same flow analysis applies.

Both generators output a complete `defmodule` with `deflayer` declarations
that you can paste into your project and customize.

## Available layers

### Linear
- `ExTorch.NN.Linear` -- `:in_features`, `:out_features`, `:bias`

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
- `ExTorch.NN.Dropout` -- `:p`, `:inplace`
- `ExTorch.NN.Embedding` -- `:num_embeddings`, `:embedding_dim`, `:padding_idx`
- `ExTorch.NN.Flatten` -- `:start_dim`, `:end_dim`
- `ExTorch.NN.Unflatten` -- `:dim`, `:sizes`
