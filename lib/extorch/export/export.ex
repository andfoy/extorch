defmodule ExTorch.Export do
  @moduledoc """
  Read and introspect PyTorch ExportedProgram `.pt2` archives.

  This module provides a pure-Elixir reader for `.pt2` files produced by
  `torch.export.save()`. It can extract the model graph, weight metadata,
  and raw weight tensors without requiring Python or C++ ExportedProgram support.

  ## Python export workflow

      import torch

      model = MyModel()
      model.eval()
      exported = torch.export.export(model, (example_input,))
      torch.export.save(exported, "model.pt2")

  ## Elixir usage

      # Load and run inference directly
      model = ExTorch.Export.load("model.pt2")
      output = ExTorch.Export.forward(model, [input])

      # Or read schema and weights separately
      schema = ExTorch.Export.read_schema("model.pt2")
      weights = ExTorch.Export.read_weights("model.pt2")

      # Generate DSL source code
      IO.puts(ExTorch.Export.to_elixir("model.pt2", "MyModel"))

  ## Note

  This reads `.pt2` files from `torch.export.save`, NOT from
  `aoti_compile_and_package`. AOTI-compiled `.pt2` files don't contain
  the graph or separable weights -- use `ExTorch.AOTI` for those.
  """

  defmodule Model do
    @moduledoc """
    A loaded ExportedProgram model ready for inference.

    Contains the computation graph, weight tensors, and input/output mappings.
    """

    @type t :: %__MODULE__{
            schema: map(),
            weights: %{String.t() => ExTorch.Tensor.t()},
            param_inputs: [String.t()],
            user_inputs: [String.t()]
          }

    defstruct [:schema, :weights, :param_inputs, :user_inputs]
  end

  @doc """
  Load an exported `.pt2` model for inference.

  Reads the graph and weights, and prepares the model for `forward/2`.

  ## Args
    * `path` (`String`) - path to the `.pt2` file from `torch.export.save`.

  ## Returns
  An `%ExTorch.Export.Model{}` struct.

  ## Example

      model = ExTorch.Export.load("model.pt2")
      output = ExTorch.Export.forward(model, [input_tensor])
  """
  @spec load(String.t()) :: Model.t()
  def load(path) do
    schema = read_schema(path)
    weights = read_weights(path)

    # Separate parameter/buffer inputs (p_* and b_*) from user inputs
    {param_inputs, user_inputs} =
      Enum.split_with(schema.inputs, fn name ->
        String.starts_with?(name, "p_") or String.starts_with?(name, "b_")
      end)

    %Model{
      schema: schema,
      weights: weights,
      param_inputs: param_inputs,
      user_inputs: user_inputs
    }
  end

  @doc """
  Run inference on a loaded Export model.

  Interprets the ATen computation graph, dispatching each operation to
  the corresponding ExTorch tensor function.

  ## Args
    * `model` (`ExTorch.Export.Model`) - the loaded model.
    * `inputs` (`[ExTorch.Tensor]`) - input tensors, matching the model's user inputs.

  ## Returns
  The output tensor (or list of tensors for multi-output models).

  ## Example

      model = ExTorch.Export.load("model.pt2")
      input = ExTorch.randn({1, 10})
      output = ExTorch.Export.forward(model, [input])
  """
  @spec forward(Model.t(), [ExTorch.Tensor.t()]) :: ExTorch.Tensor.t() | [ExTorch.Tensor.t()]
  def forward(%Model{} = model, inputs) when is_list(inputs) do
    # Build a lookup from graph ref names to weight FQNs
    weight_fqns = Map.keys(model.weights)

    # Build initial value map: parameters + user inputs
    values =
      model.param_inputs
      |> Enum.reduce(%{}, fn param_name, acc ->
        fqn = find_matching_fqn(param_name, weight_fqns)
        case fqn do
          nil -> acc  # Buffer not in weights (e.g., num_batches_tracked)
          fqn -> Map.put(acc, param_name, Map.fetch!(model.weights, fqn))
        end
      end)

    values =
      model.user_inputs
      |> Enum.zip(inputs)
      |> Enum.reduce(values, fn {name, tensor}, acc ->
        Map.put(acc, name, tensor)
      end)

    # Execute graph nodes in order
    values =
      Enum.reduce(model.schema.graph, values, fn node, acc ->
        output_tensor = execute_node(node, acc)
        output_name = List.first(node.outputs) || "unknown"
        Map.put(acc, output_name, output_tensor)
      end)

    # Return output(s)
    output_names = model.schema.outputs
    outputs = Enum.map(output_names, &Map.fetch!(values, &1))

    case outputs do
      [single] -> single
      multiple -> multiple
    end
  end

  # Match a graph ref name like "p_attn_out_proj_weight" to a weight FQN
  # like "attn.out_proj.weight".
  #
  # The graph uses p_ prefix + dots replaced with underscores. Since FQNs
  # can contain underscores themselves (e.g., "out_proj"), we reverse the
  # mapping: for each known FQN, compute what the graph ref would be, and
  # build a lookup table.
  defp find_matching_fqn(ref, fqns) do
    stripped =
      cond do
        String.starts_with?(ref, "p_") -> String.trim_leading(ref, "p_")
        String.starts_with?(ref, "b_") -> String.trim_leading(ref, "b_")
        true -> ref
      end

    # FQN "attn.out_proj.weight" → graph ref "attn_out_proj_weight"
    Enum.find(fqns, fn fqn ->
      String.replace(fqn, ".", "_") == stripped
    end)
  end

  # ============================================================================
  # Graph interpreter: ATen op dispatch
  # ============================================================================

  defp execute_node(node, values) do
    i = node.inputs
    case node.target do
      # ==== Linear algebra ====
      "torch.ops.aten.linear.default" ->
        result = ExTorch.matmul(resolve(i, "input", values), ExTorch.transpose(resolve(i, "weight", values), 0, 1))
        case resolve_optional(i, "bias", values) do
          nil -> result
          bias -> ExTorch.add(result, bias)
        end

      "torch.ops.aten.mm.default" -> ExTorch.mm(resolve(i, "self", values), resolve(i, "mat2", values))
      "torch.ops.aten.bmm.default" -> ExTorch.bmm(resolve(i, "self", values), resolve(i, "mat2", values))

      "torch.ops.aten.addmm.default" ->
        input = resolve(i, "self", values)
        mat1 = resolve(i, "mat1", values)
        mat2 = resolve(i, "mat2", values)
        ExTorch.add(input, ExTorch.mm(mat1, mat2))

      # ==== Activations ====
      "torch.ops.aten.relu.default" -> ExTorch.functional_relu(resolve(i, "self", values))
      "torch.ops.aten.relu_.default" -> ExTorch.functional_relu(resolve(i, "self", values))
      "torch.ops.aten.gelu.default" -> ExTorch.NN.forward(resolve(i, "self", values), ExTorch.NN.gelu())
      "torch.ops.aten.sigmoid.default" -> ExTorch.NN.forward(resolve(i, "self", values), ExTorch.NN.sigmoid())
      "torch.ops.aten.tanh.default" -> ExTorch.NN.forward(resolve(i, "self", values), ExTorch.NN.tanh())
      "torch.ops.aten.silu.default" -> ExTorch.NN.forward(resolve(i, "self", values), ExTorch.NN.silu())

      "torch.ops.aten.elu.default" ->
        ExTorch.NN.forward(resolve(i, "self", values),
          ExTorch.NN.elu(alpha: resolve_float(i, "alpha", 1.0)))

      "torch.ops.aten.leaky_relu.default" ->
        ExTorch.NN.forward(resolve(i, "self", values),
          ExTorch.NN.leaky_relu(negative_slope: resolve_float(i, "negative_slope", 0.01)))

      "torch.ops.aten.hardtanh.default" ->
        ExTorch.clamp(resolve(i, "self", values),
          resolve_float(i, "min_val", -1.0), resolve_float(i, "max_val", 1.0))
      "torch.ops.aten.hardtanh_.default" ->
        ExTorch.clamp(resolve(i, "self", values),
          resolve_float(i, "min_val", -1.0), resolve_float(i, "max_val", 1.0))

      # ==== Softmax ====
      "torch.ops.aten._softmax.default" -> ExTorch.functional_softmax(resolve(i, "self", values), resolve_int(i, "dim", -1))
      "torch.ops.aten.softmax.int" -> ExTorch.functional_softmax(resolve(i, "self", values), resolve_int(i, "dim", -1))
      "torch.ops.aten._log_softmax.default" -> ExTorch.functional_log_softmax(resolve(i, "self", values), resolve_int(i, "dim", -1))

      # ==== Binary arithmetic (Tensor) ====
      "torch.ops.aten.add.Tensor" -> binary_op(i, values, &ExTorch.add/2)
      "torch.ops.aten.add_.Tensor" -> binary_op(i, values, &ExTorch.add/2)
      "torch.ops.aten.sub.Tensor" -> binary_op(i, values, &ExTorch.sub/2)
      "torch.ops.aten.mul.Tensor" -> binary_op(i, values, &ExTorch.mul/2)
      "torch.ops.aten.div.Tensor" -> binary_op(i, values, &ExTorch.tensor_div/2)
      "torch.ops.aten.pow.Tensor_Scalar" -> ExTorch.pow_tensor(resolve(i, "self", values), resolve_float(i, "exponent", 2.0))

      # ==== Unary math ====
      "torch.ops.aten.abs.default" -> ExTorch.tensor_abs(resolve(i, "self", values))
      "torch.ops.aten.neg.default" -> ExTorch.neg(resolve(i, "self", values))
      "torch.ops.aten.exp.default" -> ExTorch.tensor_exp(resolve(i, "self", values))
      "torch.ops.aten.log.default" -> ExTorch.tensor_log(resolve(i, "self", values))
      "torch.ops.aten.sqrt.default" -> ExTorch.tensor_sqrt(resolve(i, "self", values))
      "torch.ops.aten.rsqrt.default" -> ExTorch.tensor_div(ExTorch.ones(resolve(i, "self", values).size), ExTorch.tensor_sqrt(resolve(i, "self", values)))
      "torch.ops.aten.sin.default" -> ExTorch.tensor_sin(resolve(i, "self", values))
      "torch.ops.aten.cos.default" -> ExTorch.tensor_cos(resolve(i, "self", values))
      "torch.ops.aten.reciprocal.default" -> ExTorch.tensor_div(ExTorch.ones(resolve(i, "self", values).size), resolve(i, "self", values))

      # ==== Clamping ====
      "torch.ops.aten.clamp.default" ->
        ExTorch.clamp(resolve(i, "self", values), resolve_float(i, "min", -1.0e30), resolve_float(i, "max", 1.0e30))
      "torch.ops.aten.maximum.default" -> ExTorch.maximum(resolve(i, "self", values), resolve(i, "other", values))
      "torch.ops.aten.minimum.default" -> ExTorch.minimum(resolve(i, "self", values), resolve(i, "other", values))

      # ==== Comparisons ====
      "torch.ops.aten.eq.Tensor" -> ExTorch.eq(resolve(i, "self", values), resolve(i, "other", values))
      "torch.ops.aten.ne.Tensor" -> ExTorch.ne(resolve(i, "self", values), resolve(i, "other", values))
      "torch.ops.aten.gt.Tensor" -> ExTorch.gt(resolve(i, "self", values), resolve(i, "other", values))
      "torch.ops.aten.ge.Tensor" -> ExTorch.ge(resolve(i, "self", values), resolve(i, "other", values))
      "torch.ops.aten.lt.Tensor" -> ExTorch.lt(resolve(i, "self", values), resolve(i, "other", values))
      "torch.ops.aten.le.Tensor" -> ExTorch.le(resolve(i, "self", values), resolve(i, "other", values))

      # ==== Reductions ====
      "torch.ops.aten.sum.dim_IntList" ->
        ExTorch.sum(resolve(i, "self", values), resolve_int(i, "dim", 0), resolve_int(i, "keepdim", 0) == 1)
      "torch.ops.aten.mean.dim" ->
        dim = case List.keyfind(i, "dim", 0) do
          {"dim", {:int, v}} -> v
          {"dim", {:raw, %{"as_ints" => [v]}}} -> v
          {"dim", {:raw, [%{"as_int" => v}]}} -> v
          _ -> 0
        end
        keepdim = resolve_int(i, "keepdim", 0) == 1
        ExTorch.mean(resolve(i, "self", values), dim, keepdim)
      "torch.ops.aten.amax.default" ->
        ExTorch.amax(resolve(i, "self", values), resolve_int(i, "dim", 0), resolve_int(i, "keepdim", 0) == 1)
      "torch.ops.aten.amin.default" ->
        ExTorch.amin(resolve(i, "self", values), resolve_int(i, "dim", 0), resolve_int(i, "keepdim", 0) == 1)
      "torch.ops.aten.argmax.default" ->
        ExTorch.argmax(resolve(i, "self", values))
      "torch.ops.aten.argmin.default" ->
        ExTorch.argmin(resolve(i, "self", values))

      # ==== Conditional / masking ====
      "torch.ops.aten.where.self" ->
        ExTorch.tensor_where(resolve(i, "condition", values), resolve(i, "self", values), resolve(i, "other", values))
      "torch.ops.aten.masked_fill.Scalar" ->
        ExTorch.masked_fill(resolve(i, "self", values), resolve(i, "mask", values), resolve_float(i, "value", 0.0))

      # ==== Shape / layout ====
      "torch.ops.aten.view.default" ->
        ExTorch.view(resolve(i, "self", values), List.to_tuple(resolve_int_list_flex(i, "size", [])))
      "torch.ops.aten.reshape.default" ->
        ExTorch.reshape(resolve(i, "self", values), List.to_tuple(resolve_int_list_flex(i, "shape", [])))
      "torch.ops.aten.expand.default" ->
        ExTorch.expand(resolve(i, "self", values), List.to_tuple(resolve_int_list_flex(i, "size", [])))
      "torch.ops.aten.permute.default" ->
        ExTorch.permute(resolve(i, "self", values), List.to_tuple(resolve_int_list_flex(i, "dims", [])))
      "torch.ops.aten.transpose.int" ->
        ExTorch.transpose(resolve(i, "self", values), resolve_int(i, "dim0", 0), resolve_int(i, "dim1", 1))
      "torch.ops.aten.unsqueeze.default" ->
        ExTorch.unsqueeze(resolve(i, "self", values), resolve_int(i, "dim", 0))
      "torch.ops.aten.squeeze.dim" ->
        ExTorch.squeeze(resolve(i, "self", values), [resolve_int(i, "dim", 0)])
      "torch.ops.aten.squeeze.dims" ->
        ExTorch.squeeze(resolve(i, "self", values), resolve_int_list_flex(i, "dim", [0]))
      "torch.ops.aten.cat.default" ->
        tensors = resolve_tensor_list(i, "tensors", values)
        ExTorch.cat(tensors, resolve_int(i, "dim", 0))
      "torch.ops.aten.unflatten.int" ->
        input = resolve(i, "self", values)
        dim = resolve_int(i, "dim", 0)
        sizes = resolve_int_list_flex(i, "sizes", [])
        ExTorch.NN.forward(input, ExTorch.NN.unflatten(dim, sizes))
      "torch.ops.aten.select.int" ->
        ExTorch.select(resolve(i, "self", values), resolve_int(i, "dim", 0), resolve_int(i, "index", 0))
      "torch.ops.aten.flatten.using_ints" ->
        ExTorch.NN.forward(resolve(i, "self", values),
          ExTorch.NN.flatten(start_dim: resolve_int(i, "start_dim", 1), end_dim: resolve_int(i, "end_dim", -1)))

      # ==== Tensor creation / manipulation ====
      "torch.ops.aten.clone.default" -> ExTorch.clone(resolve(i, "self", values))
      "torch.ops.aten._to_copy.default" -> ExTorch.clone(resolve(i, "self", values))
      "torch.ops.aten.alias.default" -> resolve(i, "self", values)
      "torch.ops.aten.contiguous.default" -> ExTorch.contiguous(resolve(i, "self", values))
      "torch.ops.aten.detach.default" -> ExTorch.detach(resolve(i, "self", values))

      # ==== Convolution (generic) ====
      "torch.ops.aten.convolution.default" ->
        execute_convolution(i, values)
      "torch.ops.aten.conv2d.default" ->
        execute_conv2d(i, values)

      # ==== Normalization ====
      "torch.ops.aten.batch_norm.default" ->
        execute_batch_norm(i, values)
      "torch.ops.aten._native_batch_norm_legit_no_training.default" ->
        execute_batch_norm(i, values)
      "torch.ops.aten.native_layer_norm.default" ->
        execute_layer_norm(i, values)
      "torch.ops.aten.layer_norm.default" ->
        execute_layer_norm(i, values)
      "torch.ops.aten.native_group_norm.default" ->
        execute_group_norm(i, values)

      # ==== Pooling ====
      "torch.ops.aten._adaptive_avg_pool2d.default" ->
        output_size = resolve_int_list_flex(i, "output_size", [1, 1])
        ExTorch.NN.forward(resolve(i, "self", values), ExTorch.NN.adaptive_avg_pool2d(Enum.at(output_size, 0), Enum.at(output_size, 1)))
      "torch.ops.aten.adaptive_avg_pool2d.default" ->
        output_size = resolve_int_list_flex(i, "output_size", [1, 1])
        ExTorch.NN.forward(resolve(i, "self", values), ExTorch.NN.adaptive_avg_pool2d(Enum.at(output_size, 0), Enum.at(output_size, 1)))
      "torch.ops.aten.max_pool2d.default" ->
        ExTorch.NN.forward(resolve(i, "self", values),
          ExTorch.NN.max_pool2d(hd(resolve_int_list_flex(i, "kernel_size", [2])),
            stride: hd(resolve_int_list_flex(i, "stride", [2])), padding: hd(resolve_int_list_flex(i, "padding", [0]))))
      "torch.ops.aten.avg_pool2d.default" ->
        ExTorch.NN.forward(resolve(i, "self", values),
          ExTorch.NN.avg_pool2d(hd(resolve_int_list_flex(i, "kernel_size", [2])),
            stride: hd(resolve_int_list_flex(i, "stride", [2])), padding: hd(resolve_int_list_flex(i, "padding", [0]))))
      "torch.ops.aten.max_pool2d_with_indices.default" ->
        ExTorch.NN.forward(resolve(i, "self", values),
          ExTorch.NN.max_pool2d(hd(resolve_int_list_flex(i, "kernel_size", [2])),
            stride: hd(resolve_int_list_flex(i, "stride", [2])), padding: hd(resolve_int_list_flex(i, "padding", [0]))))

      # ==== Dropout (identity in eval mode) ====
      "torch.ops.aten.native_dropout.default" -> resolve(i, "input", values)
      "torch.ops.aten.dropout.default" -> resolve(i, "input", values)

      # ==== Embedding ====
      "torch.ops.aten.embedding.default" ->
        weight = resolve(i, "weight", values)
        indices = resolve(i, "indices", values)
        ExTorch.index_select(weight, 0, ExTorch.reshape(indices, {ExTorch.Native.numel(indices)}))
        |> then(&ExTorch.view(&1, Tuple.append(indices.size, elem(weight.size, 1))))

      # ==== Gather / scatter / index ====
      "torch.ops.aten.gather.default" ->
        ExTorch.gather(resolve(i, "self", values), resolve_int(i, "dim", 0), resolve(i, "index", values))
      "torch.ops.aten.scatter.src" ->
        ExTorch.scatter(resolve(i, "self", values), resolve_int(i, "dim", 0), resolve(i, "index", values), resolve(i, "src", values))
      "torch.ops.aten.index_select.default" ->
        ExTorch.index_select(resolve(i, "self", values), resolve_int(i, "dim", 0), resolve(i, "index", values))
      "torch.ops.aten.nonzero.default" ->
        ExTorch.nonzero(resolve(i, "self", values))
      "torch.ops.aten.sort.default" ->
        ExTorch.sort(resolve(i, "self", values))
      "torch.ops.aten.topk.default" ->
        ExTorch.topk(resolve(i, "self", values), resolve_int(i, "k", 1))

      other ->
        raise "Unsupported ATen op: #{other}. Consider using ExTorch.AOTI for compiled inference."
    end
  end

  # Complex op helpers

  defp execute_conv2d(i, values) do
    input = resolve(i, "input", values)
    weight = resolve(i, "weight", values)
    bias = resolve_optional(i, "bias", values)
    stride = resolve_int_list_flex(i, "stride", [1, 1])
    padding = resolve_int_list_flex(i, "padding", [0, 0])
    dilation = resolve_int_list_flex(i, "dilation", [1, 1])
    groups = resolve_int(i, "groups", 1)
    [out_ch, in_ch, kh, _kw] = Tuple.to_list(weight.size)
    layer = ExTorch.NN.conv2d(in_ch * groups, out_ch, kh,
      stride: hd(stride), padding: hd(padding), dilation: hd(dilation), groups: groups)
    params = [{"weight", weight}] ++ if(bias, do: [{"bias", bias}], else: [])
    ExTorch.NN.copy_parameters(layer, params)
    ExTorch.NN.forward(input, layer)
  end

  defp execute_convolution(i, values) do
    input = resolve(i, "input", values)
    weight = resolve(i, "weight", values)
    bias = resolve_optional(i, "bias", values)
    stride = resolve_int_list_flex(i, "stride", [1])
    padding = resolve_int_list_flex(i, "padding", [0])
    dilation = resolve_int_list_flex(i, "dilation", [1])
    groups = resolve_int(i, "groups", 1)
    ndim = tuple_size(weight.size) - 2
    weight_dims = Tuple.to_list(weight.size)
    out_ch = hd(weight_dims)
    in_ch = Enum.at(weight_dims, 1)
    k = Enum.at(weight_dims, 2)
    layer = case ndim do
      1 -> ExTorch.NN.conv1d(in_ch * groups, out_ch, k, stride: hd(stride), padding: hd(padding), dilation: hd(dilation), groups: groups)
      2 -> ExTorch.NN.conv2d(in_ch * groups, out_ch, k, stride: hd(stride), padding: hd(padding), dilation: hd(dilation), groups: groups)
      3 -> ExTorch.NN.conv3d(in_ch * groups, out_ch, k, stride: hd(stride), padding: hd(padding), dilation: hd(dilation), groups: groups)
    end
    params = [{"weight", weight}] ++ if(bias, do: [{"bias", bias}], else: [])
    ExTorch.NN.copy_parameters(layer, params)
    ExTorch.NN.forward(input, layer)
  end

  defp execute_batch_norm(i, values) do
    input = resolve(i, "input", values)
    weight = resolve_optional(i, "weight", values)
    bias = resolve_optional(i, "bias", values)
    running_mean = resolve_optional(i, "running_mean", values)
    running_var = resolve_optional(i, "running_var", values)
    num_features = elem(input.size, 1)
    layer = ExTorch.NN.batch_norm2d(num_features)
    ExTorch.NN.eval(layer)
    params = []
    params = if weight, do: params ++ [{"weight", weight}], else: params
    params = if bias, do: params ++ [{"bias", bias}], else: params
    params = if running_mean, do: params ++ [{"running_mean", running_mean}], else: params
    params = if running_var, do: params ++ [{"running_var", running_var}], else: params
    if params != [], do: ExTorch.NN.copy_parameters(layer, params)
    ExTorch.NN.forward(input, layer)
  end

  defp execute_layer_norm(i, values) do
    input = resolve_any(i, ["input", "self"], values)
    weight = resolve_optional(i, "weight", values)
    bias = resolve_optional(i, "bias", values)
    normalized_shape = resolve_int_list_flex(i, "normalized_shape", [elem(input.size, tuple_size(input.size) - 1)])
    layer = ExTorch.NN.layer_norm(normalized_shape)
    params = []
    params = if weight, do: params ++ [{"weight", weight}], else: params
    params = if bias, do: params ++ [{"bias", bias}], else: params
    if params != [], do: ExTorch.NN.copy_parameters(layer, params)
    ExTorch.NN.forward(input, layer)
  end

  defp execute_group_norm(i, values) do
    input = resolve(i, "input", values)
    weight = resolve_optional(i, "weight", values)
    bias = resolve_optional(i, "bias", values)
    num_groups = resolve_int(i, "group", 1)
    num_channels = elem(input.size, 1)
    layer = ExTorch.NN.group_norm(num_groups, num_channels)
    params = []
    params = if weight, do: params ++ [{"weight", weight}], else: params
    params = if bias, do: params ++ [{"bias", bias}], else: params
    if params != [], do: ExTorch.NN.copy_parameters(layer, params)
    ExTorch.NN.forward(input, layer)
  end

  defp resolve(inputs, name, values) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:tensor, ref}} -> Map.fetch!(values, ref)
      _ -> raise "Missing input '#{name}'"
    end
  end

  defp resolve_optional(inputs, name, values) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:tensor, ref}} -> Map.get(values, ref)
      {^name, :none} -> nil
      nil -> nil
      _ -> nil
    end
  end

  # Handle binary ops where 'other' can be either a tensor or a scalar
  defp binary_op(inputs, values, op) do
    self = resolve(inputs, "self", values)
    other = resolve_tensor_or_scalar(inputs, "other", values, self)
    op.(self, other)
  end

  defp resolve_tensor_or_scalar(inputs, name, values, ref_tensor) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:tensor, ref}} -> Map.fetch!(values, ref)
      {^name, {:float, val}} -> ExTorch.full(ref_tensor.size, val)
      {^name, {:int, val}} -> ExTorch.full(ref_tensor.size, val)
      _ -> raise "Missing input '#{name}'"
    end
  end

  defp resolve_any(inputs, names, values) do
    Enum.find_value(names, fn name ->
      case List.keyfind(inputs, name, 0) do
        {^name, {:tensor, ref}} -> Map.get(values, ref)
        _ -> nil
      end
    end) || raise "Missing input (tried #{inspect(names)})"
  end

  # Handles both {:raw, [%{"as_int" => v}]} and {:raw, %{"as_ints" => [...]}}
  defp resolve_int_list_flex(inputs, name, default) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:raw, list}} when is_list(list) ->
        Enum.map(list, fn
          %{"as_int" => v} -> v
          v when is_integer(v) -> v
        end)
      {^name, {:raw, %{"as_ints" => list}}} when is_list(list) ->
        Enum.map(list, fn v when is_integer(v) -> v end)
      _ -> default
    end
  end

  defp resolve_float(inputs, name, default) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:float, val}} -> val
      {^name, {:int, val}} -> val / 1
      _ -> default
    end
  end

  defp resolve_tensor_list(inputs, name, values) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:raw, list}} when is_list(list) ->
        Enum.map(list, fn
          %{"as_tensor" => %{"name" => ref}} -> Map.fetch!(values, ref)
          {:tensor, ref} -> Map.fetch!(values, ref)
        end)
      {^name, {:raw, %{"as_tensors" => list}}} when is_list(list) ->
        Enum.map(list, fn
          %{"name" => ref} -> Map.fetch!(values, ref)
        end)
      _ -> []
    end
  end

  defp resolve_int(inputs, name, default) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:int, val}} -> val
      _ -> default
    end
  end

  # PyTorch dtype enum values → ExTorch dtype atoms
  @dtype_map %{
    0 => :uint8,
    1 => :int8,
    2 => :int16,
    3 => :int32,
    4 => :int64,
    5 => :float16,
    6 => :float32,
    7 => :float32,   # torch.float is 7 in some versions
    8 => :float64,
    9 => :complex32,
    10 => :complex64,
    11 => :complex128,
    12 => :bool,
    15 => :bfloat16
  }

  @doc """
  Read the model schema from an exported `.pt2` archive.

  Returns a map with:
    * `:graph` - the computation graph as a list of node maps
    * `:inputs` - graph input names
    * `:outputs` - graph output names
    * `:weights` - weight metadata (name → shape, dtype, requires_grad)
  """
  @spec read_schema(String.t()) :: map()
  def read_schema(path) do
    archive = read_archive(path)
    model_name = detect_model_name(archive)

    graph_data =
      read_json(archive, "#{model_name}/models/#{model_name}.json")
      |> case do
        empty when map_size(empty) == 0 ->
          read_json(archive, "#{model_name}/models/model.json")
        found -> found
      end
    # The weights config file may use either the archive name or "model" as prefix
    weights_config =
      read_json(archive, "#{model_name}/data/weights/#{model_name}_weights_config.json")
      |> case do
        empty when map_size(empty) == 0 ->
          read_json(archive, "#{model_name}/data/weights/model_weights_config.json")
        found -> found
      end

    graph_module = graph_data["graph_module"]["graph"]

    inputs =
      Enum.map(graph_module["inputs"] || [], fn input ->
        get_in(input, ["as_tensor", "name"]) || "unknown"
      end)

    outputs =
      Enum.map(graph_module["outputs"] || [], fn output ->
        get_in(output, ["as_tensor", "name"]) || "unknown"
      end)

    nodes =
      Enum.map(graph_module["nodes"] || [], fn node ->
        %{
          target: node["target"],
          inputs: parse_node_inputs(node["inputs"] || []),
          outputs: parse_node_outputs(node["outputs"] || [])
        }
      end)

    weights =
      for {fqn, config} <- weights_config["config"] || %{}, into: %{} do
        meta = config["tensor_meta"]
        sizes = Enum.map(meta["sizes"] || [], fn s -> s["as_int"] end)
        dtype_int = meta["dtype"] || 6

        {fqn, %{
          shape: sizes,
          dtype: Map.get(@dtype_map, dtype_int, :float32),
          requires_grad: meta["requires_grad"] || false,
          file: config["path_name"]
        }}
      end

    %{
      graph: nodes,
      inputs: inputs,
      outputs: outputs,
      weights: weights,
      model_name: model_name
    }
  end

  @doc """
  Load weight tensors from an exported `.pt2` archive.

  Returns a map of `%{fqn => %ExTorch.Tensor{}}`.
  """
  @spec read_weights(String.t()) :: %{String.t() => ExTorch.Tensor.t()}
  def read_weights(path) do
    archive = read_archive(path)
    schema = read_schema(path)

    for {fqn, meta} <- schema.weights, into: %{} do
      weight_path = "#{schema.model_name}/data/weights/#{meta.file}"
      binary = read_file(archive, weight_path)

      shape = List.to_tuple(meta.shape)
      tensor = ExTorch.Native.from_binary(binary, shape, meta.dtype)

      {fqn, tensor}
    end
  end

  @doc """
  Generate an `ExTorch.NN.Module` DSL definition from an exported `.pt2` archive.

  Maps ATen operations in the graph to ExTorch NN layer types where possible.

  ## Args
    * `path` - path to the `.pt2` file.
    * `module_name` - name for the generated Elixir module.
  """
  @spec to_elixir(String.t(), String.t()) :: String.t()
  def to_elixir(path, module_name \\ "MyModel") do
    schema = read_schema(path)
    generate_elixir_source(schema, module_name)
  end

  # ============================================================================
  # Private: Archive reading
  # ============================================================================

  defp read_archive(path) do
    {:ok, files} = :zip.extract(String.to_charlist(path), [:memory])
    for {name, data} <- files, into: %{} do
      {List.to_string(name), data}
    end
  end

  defp detect_model_name(archive) do
    archive
    |> Map.keys()
    |> Enum.find_value(fn key ->
      case String.split(key, "/", parts: 2) do
        [name, _rest] -> name
        _ -> nil
      end
    end)
  end

  defp read_json(archive, key) do
    case Map.get(archive, key) do
      nil -> %{}
      data -> Jason.decode!(data)
    end
  end

  defp read_file(archive, key) do
    Map.fetch!(archive, key)
  end

  # ============================================================================
  # Private: Graph parsing
  # ============================================================================

  defp parse_node_inputs(inputs) do
    Enum.map(inputs, fn input ->
      name = input["name"]
      arg = input["arg"]

      value =
        cond do
          is_map(arg) and arg["as_tensor"] -> {:tensor, arg["as_tensor"]["name"]}
          is_map(arg) and arg["as_int"] != nil -> {:int, arg["as_int"]}
          is_map(arg) and arg["as_float"] != nil -> {:float, arg["as_float"]}
          is_map(arg) and arg["as_bool"] != nil -> {:bool, arg["as_bool"]}
          is_map(arg) and arg["as_none"] != nil -> :none
          is_list(arg) -> {:raw, arg}
          true -> {:raw, arg}
        end

      {name, value}
    end)
  end

  defp parse_node_outputs(outputs) do
    Enum.map(outputs, fn output ->
      cond do
        output["as_tensor"] -> output["as_tensor"]["name"]
        true -> "unknown"
      end
    end)
  end

  # ============================================================================
  # Private: DSL generation
  # ============================================================================

  defp generate_elixir_source(schema, module_name) do
    layers = infer_layers(schema)

    layer_lines =
      layers
      |> Enum.map(fn {name, type, opts} ->
        opts_str = if opts == "", do: "", else: ", #{opts}"
        "  deflayer :#{name}, #{type}#{opts_str}"
      end)
      |> Enum.join("\n")

    forward_body = generate_forward_body(schema)

    """
    defmodule #{module_name} do
      use ExTorch.NN.Module

    #{layer_lines}

      def forward(model, x) do
    #{forward_body}
      end
    end
    """
  end

  # Generate a forward body with proper variable assignments for branching data flow.
  defp generate_forward_body(schema) do
    graph = schema.graph
    # Find the user input name (the non-parameter input)
    user_inputs = schema.inputs |> Enum.reject(&(String.starts_with?(&1, "p_") or String.starts_with?(&1, "b_")))
    user_input = List.first(user_inputs) || "x"

    # Count how many times each value is referenced as an input
    ref_counts = count_references(graph)

    # Walk nodes and generate code lines
    {lines, _} =
      Enum.reduce(graph, {[], %{user_input => "x"}}, fn node, {lines_acc, var_map} ->
        output_name = List.first(node.outputs) || "unknown"

        # Get the primary tensor input (self or input)
        primary_ref = get_primary_ref(node)
        _primary_var = if primary_ref, do: Map.get(var_map, primary_ref, sanitize_var(primary_ref))

        # Generate the expression for this node
        expr = node_to_elixir(node, var_map, schema.weights)

        # Decide if we need a named variable (value used more than once downstream)
        needs_binding = Map.get(ref_counts, output_name, 0) > 1
        var_name = sanitize_var(output_name)

        {new_lines, new_var_map} =
          if needs_binding do
            {lines_acc ++ ["    #{var_name} = #{expr}"], Map.put(var_map, output_name, var_name)}
          else
            {lines_acc ++ ["    #{var_name} = #{expr}"], Map.put(var_map, output_name, var_name)}
          end

        {new_lines, new_var_map}
      end)

    # The last line's variable is the return value
    case lines do
      [] -> "    x"
      _ -> Enum.join(lines, "\n")
    end
  end

  defp count_references(graph) do
    Enum.reduce(graph, %{}, fn node, acc ->
      tensor_refs =
        node.inputs
        |> Enum.flat_map(fn
          {_name, {:tensor, ref}} -> [ref]
          _ -> []
        end)

      Enum.reduce(tensor_refs, acc, fn ref, inner_acc ->
        Map.update(inner_acc, ref, 1, &(&1 + 1))
      end)
    end)
  end

  defp get_primary_ref(node) do
    # The primary tensor input -- typically "self" or "input"
    Enum.find_value(node.inputs, fn
      {"self", {:tensor, ref}} -> ref
      {"input", {:tensor, ref}} -> ref
      _ -> nil
    end)
  end

  defp node_to_elixir(node, var_map, weights) do
    i = node.inputs

    resolve_var = fn name ->
      case List.keyfind(i, name, 0) do
        {^name, {:tensor, ref}} -> Map.get(var_map, ref, sanitize_var(ref))
        _ -> "nil"
      end
    end

    case node.target do
      "torch.ops.aten.linear.default" ->
        input_v = resolve_var.("input")
        "#{input_v} |> layer(model, :#{layer_name_from_weight(i, weights, "linear")})"

      t when t in ["torch.ops.aten.relu.default", "torch.ops.aten.relu_.default"] ->
        "#{resolve_var.("self")} |> layer(model, :#{sanitize_name(List.first(node.outputs) || "relu")})"

      t when t in ["torch.ops.aten.add.Tensor", "torch.ops.aten.add_.Tensor"] ->
        "ExTorch.add(#{resolve_var.("self")}, #{resolve_var.("other")})"

      t when t in ["torch.ops.aten.conv2d.default", "torch.ops.aten.convolution.default"] ->
        input_v = resolve_var.("input")
        "#{input_v} |> layer(model, :#{layer_name_from_weight(i, weights, "conv")})"

      "torch.ops.aten.batch_norm.default" ->
        input_v = resolve_var.("input")
        "#{input_v} |> layer(model, :#{layer_name_from_weight(i, weights, "bn")})"

      "torch.ops.aten.adaptive_avg_pool2d.default" ->
        "#{resolve_var.("self")} |> layer(model, :#{sanitize_name(List.first(node.outputs) || "pool")})"

      "torch.ops.aten.max_pool2d.default" ->
        "#{resolve_var.("self")} |> layer(model, :#{sanitize_name(List.first(node.outputs) || "maxpool")})"

      "torch.ops.aten.flatten.using_ints" ->
        "#{resolve_var.("self")} |> layer(model, :#{sanitize_name(List.first(node.outputs) || "flatten")})"

      "torch.ops.aten.gelu.default" ->
        "#{resolve_var.("self")} |> layer(model, :#{sanitize_name(List.first(node.outputs) || "gelu")})"

      "torch.ops.aten.sigmoid.default" ->
        "#{resolve_var.("self")} |> layer(model, :#{sanitize_name(List.first(node.outputs) || "sigmoid")})"

      "torch.ops.aten.tanh.default" ->
        "#{resolve_var.("self")} |> layer(model, :#{sanitize_name(List.first(node.outputs) || "tanh")})"

      "torch.ops.aten.silu.default" ->
        "#{resolve_var.("self")} |> layer(model, :#{sanitize_name(List.first(node.outputs) || "silu")})"

      "torch.ops.aten.mul.Tensor" ->
        "ExTorch.mul(#{resolve_var.("self")}, #{resolve_var.("other")})"

      "torch.ops.aten.sub.Tensor" ->
        "ExTorch.sub(#{resolve_var.("self")}, #{resolve_var.("other")})"

      "torch.ops.aten.mean.dim" ->
        "ExTorch.mean(#{resolve_var.("self")}, 1, true)"

      "torch.ops.aten.view.default" ->
        "ExTorch.view(#{resolve_var.("self")}, shape)"

      "torch.ops.aten.transpose.int" ->
        dim0 = case List.keyfind(i, "dim0", 0) do
          {"dim0", {:int, v}} -> v
          _ -> 0
        end
        dim1 = case List.keyfind(i, "dim1", 0) do
          {"dim1", {:int, v}} -> v
          _ -> 1
        end
        "ExTorch.transpose(#{resolve_var.("self")}, #{dim0}, #{dim1})"

      _ ->
        # Generic fallback -- just show the op name as a comment
        self_v = resolve_var.("self")
        if self_v != "nil" do
          "#{self_v} # #{node.target}"
        else
          resolve_var.("input") <> " # #{node.target}"
        end
    end
  end

  # Extract the layer name from a weight reference in node inputs.
  # e.g., "p_layer1_0_conv1_weight" → FQN "layer1.0.conv1.weight" → layer "layer1_0_conv1"
  defp layer_name_from_weight(inputs, weights, fallback) do
    case List.keyfind(inputs, "weight", 0) do
      {"weight", {:tensor, wref}} ->
        stripped = cond do
          String.starts_with?(wref, "p_") -> String.trim_leading(wref, "p_")
          String.starts_with?(wref, "b_") -> String.trim_leading(wref, "b_")
          true -> wref
        end
        case find_matching_fqn(stripped, Map.keys(weights)) do
          nil -> fallback
          fqn ->
            # "layer1.0.conv1.weight" → "layer1.0.conv1" → "layer1_0_conv1"
            fqn
            |> String.split(".")
            |> Enum.drop(-1)
            |> Enum.join(".")
            |> sanitize_name()
        end
      _ -> fallback
    end
  end

  defp sanitize_var(name) do
    name
    |> String.replace(".", "_")
    |> String.replace("-", "_")
    |> String.replace("~", "")
  end

  defp infer_layers(schema) do
    schema.graph
    |> Enum.map(fn node -> infer_layer_from_node(node, schema.weights) end)
    |> Enum.reject(&is_nil/1)
  end

  defp infer_layer_from_node(node, weights) do
    case node.target do
      "torch.ops.aten.linear.default" ->
        # Infer dimensions from weight tensor
        {_name, weight_ref} = Enum.find(node.inputs, fn {n, _} -> n == "weight" end) || {nil, nil}
        case weight_ref do
          {:tensor, weight_name} ->
            # weight_name is like "p_fc1_weight", FQN is "fc1.weight"
            fqn = param_ref_to_fqn(weight_name, "weight")
            case Map.get(weights, fqn) do
              %{shape: [out, inp]} ->
                layer_name = fqn |> String.split(".") |> hd() |> String.replace(".", "_")
                {layer_name, "ExTorch.NN.Linear", "in_features: #{inp}, out_features: #{out}"}
              _ ->
                nil
            end
          _ -> nil
        end

      "torch.ops.aten.relu.default" ->
        # Generate unique name from output
        output_name = List.first(node.outputs) || "relu"
        {sanitize_name(output_name), "ExTorch.NN.ReLU", ""}

      "torch.ops.aten.gelu.default" ->
        output_name = List.first(node.outputs) || "gelu"
        {sanitize_name(output_name), "ExTorch.NN.GELU", ""}

      "torch.ops.aten.sigmoid.default" ->
        output_name = List.first(node.outputs) || "sigmoid"
        {sanitize_name(output_name), "ExTorch.NN.Sigmoid", ""}

      "torch.ops.aten.tanh.default" ->
        output_name = List.first(node.outputs) || "tanh"
        {sanitize_name(output_name), "ExTorch.NN.Tanh", ""}

      "torch.ops.aten.silu.default" ->
        output_name = List.first(node.outputs) || "silu"
        {sanitize_name(output_name), "ExTorch.NN.SiLU", ""}

      _ ->
        nil
    end
  end

  # Convert "p_fc1_weight" → "fc1.weight"
  defp param_ref_to_fqn(ref_name, suffix) do
    ref_name
    |> String.trim_leading("p_")
    |> String.trim_trailing("_#{suffix}")
    |> String.replace("_", ".")
    |> Kernel.<>(".#{suffix}")
  end

  defp sanitize_name(name) do
    name |> String.replace(".", "_") |> String.replace("-", "_")
  end
end
