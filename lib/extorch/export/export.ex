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

    The `compiled_graph` field holds per-node closures built at `load/1` time
    that capture pre-resolved scalar args and tensor input refs, so `forward/2`
    is a tight loop over closures instead of re-parsing the graph each call.
    `initial_values` holds the pre-built parameter value map so forward
    doesn't re-resolve p_*/b_* names on every inference.
    """

    @type t :: %__MODULE__{
            schema: map(),
            weights: %{String.t() => ExTorch.Tensor.t()},
            param_inputs: [String.t()],
            user_inputs: [String.t()],
            compiled_graph: [{[String.t()], (map() -> any())}],
            initial_values: map(),
            device: atom() | {atom(), non_neg_integer()}
          }

    defstruct [:schema, :weights, :param_inputs, :user_inputs,
               :compiled_graph, :initial_values, device: :cpu]
  end

  @doc """
  Load an exported `.pt2` model for inference.

  Reads the graph and weights, and prepares the model for `forward/2`.

  ## Args
    * `path` (`String`) - path to the `.pt2` file from `torch.export.save`.
    * `opts` (`keyword`) - optional:
      * `:device` (`:cpu | :cuda | {:cuda, index}`) - device to place all
        weight tensors on. Defaults to `:cpu`. When set to `:cuda`, every
        loaded parameter/buffer is moved to the GPU at load time, so
        subsequent `forward/2` calls run entirely on the GPU (as long as
        the user input is also on the GPU).

  ## Returns
  An `%ExTorch.Export.Model{}` struct.

  ## Example

      # CPU (default)
      model = ExTorch.Export.load("model.pt2")
      output = ExTorch.Export.forward(model, [input_tensor])

      # GPU
      model = ExTorch.Export.load("model.pt2", device: :cuda)
      input = ExTorch.Tensor.to(cpu_input, device: :cuda)
      output = ExTorch.Export.forward(model, [input])
  """
  @spec load(String.t(), keyword()) :: Model.t()
  def load(path, opts \\ []) do
    device = Keyword.get(opts, :device, :cpu)
    schema = read_schema(path)
    weights = read_weights(path)

    # Move all weights to the target device once at load time. Forward
    # ops dispatch on tensor device, so every at::* call will run on GPU
    # when the weights (and user input) live on GPU.
    weights = maybe_move_weights(weights, device)

    # Separate parameter/buffer inputs (p_* and b_*) from user inputs
    {param_inputs, user_inputs} =
      Enum.split_with(schema.inputs, fn name ->
        String.starts_with?(name, "p_") or String.starts_with?(name, "b_")
      end)

    # Pre-build the initial parameter value map once, so forward/2 doesn't
    # re-resolve param_name -> FQN -> tensor on every inference.
    weight_fqns = Map.keys(weights)
    initial_values =
      Enum.reduce(param_inputs, %{}, fn param_name, acc ->
        case find_matching_fqn(param_name, weight_fqns) do
          nil -> acc
          fqn -> Map.put(acc, param_name, Map.fetch!(weights, fqn))
        end
      end)

    # Pre-compile each graph node into {outputs, run_closure} where the
    # closure captures scalar args and tensor input refs resolved from
    # node.inputs once. The weights are passed in so conv branches can
    # pre-pack them into MKLDNN blocked format at load time (see
    # `build_runner/2` on the "torch.ops.aten.conv2d.default" branch).
    # Non-hot ops fall through to a generic closure that
    # dispatches via the legacy execute_node/2 path.
    compiled_graph = Enum.map(schema.graph, &compile_node(&1, initial_values, device))

    %Model{
      schema: schema,
      weights: weights,
      param_inputs: param_inputs,
      user_inputs: user_inputs,
      compiled_graph: compiled_graph,
      initial_values: initial_values,
      device: device
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
    # Start from the pre-built parameter value map and splice in the user inputs.
    values =
      model.user_inputs
      |> Enum.zip(inputs)
      |> Enum.reduce(model.initial_values, fn {name, tensor}, acc ->
        Map.put(acc, name, tensor)
      end)

    # Tight loop over pre-compiled closures. Each closure already knows
    # which tensor inputs to look up and what scalar config to use.
    values =
      Enum.reduce(model.compiled_graph, values, fn {out_names, run}, acc ->
        case run.(acc) do
          tensors when is_list(tensors) ->
            Enum.zip(out_names, tensors)
            |> Enum.reduce(acc, fn {name, t}, m -> Map.put(m, name, t) end)
          tensor ->
            Map.put(acc, hd(out_names), tensor)
        end
      end)

    outputs = Enum.map(model.schema.outputs, &Map.fetch!(values, &1))

    case outputs do
      [single] -> single
      multiple -> multiple
    end
  end

  @doc """
  Run inference using the native graph executor.

  Compiles the schema graph into an instruction stream and executes the
  entire graph in a single NIF call via `execute_graph`, eliminating
  per-node NIF boundary crossings. This is significantly faster than
  `forward/2` for high-node-count models (e.g., ViT with 430 nodes)
  while still supporting all ops through the `c10::Dispatcher`.

  Falls back gracefully for ops registered via `ExTorch.Export.OpRegistry`
  since those are also dispatched through the same C++ dispatcher.

      model = ExTorch.Export.load("vit_b_16.pt2", device: :cuda)
      input = ExTorch.Tensor.to(input, device: :cuda)
      output = ExTorch.Export.forward_native(model, [input])
  """
  @spec forward_native(Model.t(), [ExTorch.Tensor.t()]) ::
          ExTorch.Tensor.t() | [ExTorch.Tensor.t()]
  def forward_native(%Model{} = model, inputs) when is_list(inputs) do
    # Build the values map: param names -> tensors
    all_names =
      Map.keys(model.initial_values) ++ model.user_inputs

    all_tensors =
      Enum.map(Map.keys(model.initial_values), &Map.fetch!(model.initial_values, &1)) ++
        inputs

    # Compile the graph into a flat instruction stream
    instructions = compile_graph_instructions(model.schema.graph)

    # Validate instructions before passing to NIF (recursive for lists)
    validate_instructions(instructions)

    result = ExTorch.Native.execute_graph(
      instructions,
      all_names,
      all_tensors,
      model.schema.outputs
    )

    case result do
      {single} when is_struct(single) -> single
      single when is_struct(single) -> single
      multiple when is_tuple(multiple) -> Tuple.to_list(multiple)
      other -> other
    end
  end

  # Compile schema graph nodes into a flat instruction stream for execute_graph.
  defp compile_graph_instructions(graph_nodes) do
    Enum.flat_map(graph_nodes, fn node ->
      {op_name, overload} = parse_op_target(node.target)
      num_args = length(node.inputs)

      header = [
        {:begin_op, op_name, num_args},
        {:overload, overload}
      ]

      outputs = Enum.map(node.outputs, &{:output, &1})

      args = Enum.map(node.inputs, fn {_name, value} ->
        encode_arg(value)
      end) |> List.flatten()

      header ++ outputs ++ args
    end)
  end

  defp parse_op_target(target) do
    # "torch.ops.aten.conv2d.default" -> {"aten::conv2d", "default"}
    case String.split(target, ".") do
      ["torch", "ops", ns, op, overload] -> {"#{ns}::#{op}", overload}
      ["torch", "ops", ns, op] -> {"#{ns}::#{op}", ""}
      # torchvision or other namespaces
      [ns, op, overload] -> {"#{ns}::#{op}", overload}
      [ns, op] -> {"#{ns}::#{op}", ""}
      _ -> {target, ""}
    end
  end

  defp encode_arg({:tensor, ref}), do: [{:ref, ref}]
  defp encode_arg({:int, v}), do: [{:int, v}]
  defp encode_arg({:float, v}), do: [{:float, v}]
  defp encode_arg({:bool, v}), do: [{:bool, v}]
  defp encode_arg(:none), do: [:none]
  defp encode_arg({:raw, %{"as_ints" => ints}}) do
    items = Enum.map(ints, &{:int, &1})
    [{:list, items}]
  end
  defp encode_arg({:raw, %{"as_floats" => floats}}) do
    items = Enum.map(floats, &{:float, &1})
    [{:list, items}]
  end
  defp encode_arg({:raw, %{"as_tensors" => tensors}}) do
    items = Enum.map(tensors, fn %{"name" => name} -> {:ref, name} end)
    [{:list, items}]
  end
  defp encode_arg({:raw, %{"as_scalar_type" => dtype}}) do
    [{:int, dtype}]
  end
  defp encode_arg({:raw, %{"as_device" => %{"type" => type, "index" => idx}}}) do
    if idx, do: [{:device, {type, idx}}], else: [{:device, type}]
  end
  defp encode_arg({:raw, nil}), do: [:none]
  defp encode_arg({:raw, v}) when is_integer(v), do: [{:int, v}]
  defp encode_arg({:raw, v}) when is_float(v), do: [{:float, v}]
  defp encode_arg({:raw, v}) when is_boolean(v), do: [{:bool, v}]
  defp encode_arg({:raw, _}), do: [:none]

  defp validate_instructions(instructions) do
    Enum.each(instructions, fn
      {:list, items} ->
        Enum.each(items, fn
          {:ref, _} -> :ok
          {:int, _} -> :ok
          {:float, _} -> :ok
          {:bool, _} -> :ok
          {:tensor, _} -> :ok
          :none -> :ok
          other -> raise "forward_native: unexpected list item #{inspect(other)}"
        end)
      {:begin_op, _, _} -> :ok
      {:overload, _} -> :ok
      {:output, _} -> :ok
      {:ref, _} -> :ok
      {:tensor, _} -> :ok
      {:int, _} -> :ok
      {:float, _} -> :ok
      {:bool, _} -> :ok
      {:device, _} -> :ok
      {:string, _} -> :ok
      :none -> :ok
      other -> raise "forward_native: unexpected instruction #{inspect(other)}"
    end)
  end

  @doc """
  Run `forward/2` with per-node timing instrumentation. Returns
  `{output, %{op_target => %{count: N, total_us: T}}}`, aggregated by op
  target so you can see which ops dominate inference time.

  Only meant for diagnostics. Adds ~1μs of measurement overhead per node
  from `:erlang.monotonic_time/1`.
  """
  @spec forward_profiled(Model.t(), [ExTorch.Tensor.t()]) ::
          {ExTorch.Tensor.t() | [ExTorch.Tensor.t()], map()}
  def forward_profiled(%Model{} = model, inputs) when is_list(inputs) do
    values =
      model.user_inputs
      |> Enum.zip(inputs)
      |> Enum.reduce(model.initial_values, fn {name, tensor}, acc ->
        Map.put(acc, name, tensor)
      end)

    # Zip compiled_graph with the original schema graph so we can read
    # node.target for aggregation (compiled_graph itself only stores the
    # closure, not the op name).
    graph_pairs = Enum.zip(model.compiled_graph, model.schema.graph)

    {values, stats} =
      Enum.reduce(graph_pairs, {values, %{}}, fn {{out_names, run}, node}, {acc, stats} ->
        t0 = :erlang.monotonic_time(:nanosecond)
        result = run.(acc)
        elapsed_ns = :erlang.monotonic_time(:nanosecond) - t0

        new_acc =
          case result do
            tensors when is_list(tensors) ->
              Enum.zip(out_names, tensors)
              |> Enum.reduce(acc, fn {name, t}, m -> Map.put(m, name, t) end)
            tensor ->
              Map.put(acc, hd(out_names), tensor)
          end

        new_stats =
          Map.update(stats, node.target, %{count: 1, total_ns: elapsed_ns}, fn e ->
            %{count: e.count + 1, total_ns: e.total_ns + elapsed_ns}
          end)

        {new_acc, new_stats}
      end)

    outputs = Enum.map(model.schema.outputs, &Map.fetch!(values, &1))
    output =
      case outputs do
        [single] -> single
        multiple -> multiple
      end

    {output, stats}
  end

  # ============================================================================
  # Graph pre-compilation (Phase C)
  #
  # At load time we walk the schema graph once and turn each node into a
  # {outputs, run_closure} pair where the closure captures:
  #   - Pre-resolved scalar args (stride, padding, eps, etc.)
  #   - Pre-resolved tensor input refs (string keys to look up in `values`)
  #
  # At forward time, the closure just does `Map.fetch!` lookups and calls the
  # right NIF. No more List.keyfind walking, no more case-dispatch on target
  # string per call. Falls through to `execute_node/2` for ops we haven't
  # specialized.
  # ============================================================================

  defp compile_node(node, initial_values, device) do
    {node.outputs, build_runner(node, initial_values, device)}
  end

  # Helpers used at compile time to extract tensor refs from node.inputs
  # without looking them up in the (not-yet-existing) values map.
  defp tensor_ref!(inputs, name) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:tensor, ref}} -> ref
      _ -> raise "Missing tensor input '#{name}'"
    end
  end

  defp tensor_ref_opt(inputs, name) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:tensor, ref}} -> ref
      _ -> nil
    end
  end

  defp tensor_ref_any(inputs, names) do
    Enum.find_value(names, fn name ->
      case List.keyfind(inputs, name, 0) do
        {^name, {:tensor, ref}} -> ref
        _ -> nil
      end
    end) || raise "Missing tensor input (tried #{inspect(names)})"
  end

  defp tensor_refs_list(inputs, name) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:raw, list}} when is_list(list) ->
        Enum.map(list, fn
          %{"as_tensor" => %{"name" => ref}} -> ref
          {:tensor, ref} -> ref
        end)
      {^name, {:raw, %{"as_tensors" => list}}} when is_list(list) ->
        Enum.map(list, fn %{"name" => ref} -> ref end)
      _ -> []
    end
  end

  # Close over a fallback that invokes the legacy dynamic dispatch.
  defp fallback_runner(node, device) do
    fn values -> execute_node(node, values, device) end
  end

  defp build_runner(node, initial_values, device) do
    i = node.inputs
    case node.target do
      # ======== Convolution ========
      "torch.ops.aten.conv2d.default" ->
        input_ref = tensor_ref!(i, "input")
        weight_ref = tensor_ref!(i, "weight")
        bias_ref = tensor_ref_opt(i, "bias")
        stride = resolve_int_list_flex(i, "stride", [1, 1])
        padding = resolve_int_list_flex(i, "padding", [0, 0])
        dilation = resolve_int_list_flex(i, "dilation", [1, 1])
        groups = resolve_int(i, "groups", 1)

        # MKLDNN weight pre-packing was explored here but regressed
        # measurably on every model in end-to-end benchmarks. at::conv2d's
        # dispatcher already picks the optimal backend and caches reorder
        # primitives after one warm call; forcing at::mkldnn_convolution
        # with a pre-packed weight skipped only negligible warm-cache
        # cost while losing at::conv2d's fast path. The pre-pack NIFs
        # (aten_mkldnn_reorder_conv2d_weight, aten_mkldnn_convolution)
        # remain available for future use (e.g. cold-start optimization,
        # per-process model warmup services).
        _ = initial_values
        fn v ->
          bias = if bias_ref, do: Map.get(v, bias_ref), else: nil
          ExTorch.Native.aten_conv2d(
            Map.fetch!(v, input_ref),
            Map.fetch!(v, weight_ref),
            bias, stride, padding, dilation, groups
          )
        end

      "torch.ops.aten.convolution.default" ->
        weight_ref = tensor_ref!(i, "weight")
        input_ref = tensor_ref!(i, "input")
        bias_ref = tensor_ref_opt(i, "bias")
        stride = resolve_int_list_flex(i, "stride", [1])
        padding = resolve_int_list_flex(i, "padding", [0])
        dilation = resolve_int_list_flex(i, "dilation", [1])
        groups = resolve_int(i, "groups", 1)
        transposed = resolve_bool(i, "transposed", false)
        # The number of spatial dims is encoded by the weight rank. We can't
        # read it from the tensor at compile time (weights are loaded), so we
        # derive it from the length of the stride/padding arg. Default 2.
        ndim = length(stride)
        output_padding = resolve_int_list_flex(i, "output_padding", List.duplicate(0, ndim))
        fn v ->
          input = Map.fetch!(v, input_ref)
          weight = Map.fetch!(v, weight_ref)
          bias = if bias_ref, do: Map.get(v, bias_ref), else: nil
          if transposed do
            ExTorch.Native.aten_convolution(
              input, weight, bias, stride, padding, dilation, true, output_padding, groups
            )
          else
            case ndim do
              1 -> ExTorch.Native.aten_conv1d(input, weight, bias, stride, padding, dilation, groups)
              2 -> ExTorch.Native.aten_conv2d(input, weight, bias, stride, padding, dilation, groups)
              3 -> ExTorch.Native.aten_conv3d(input, weight, bias, stride, padding, dilation, groups)
            end
          end
        end

      "torch.ops.aten.conv_transpose2d.input" ->
        input_ref = tensor_ref!(i, "input")
        weight_ref = tensor_ref!(i, "weight")
        bias_ref = tensor_ref_opt(i, "bias")
        stride = resolve_int_list_flex(i, "stride", [1, 1])
        padding = resolve_int_list_flex(i, "padding", [0, 0])
        output_padding = resolve_int_list_flex(i, "output_padding", [0, 0])
        dilation = resolve_int_list_flex(i, "dilation", [1, 1])
        groups = resolve_int(i, "groups", 1)
        fn v ->
          bias = if bias_ref, do: Map.get(v, bias_ref), else: nil
          ExTorch.Native.aten_conv_transpose2d(
            Map.fetch!(v, input_ref),
            Map.fetch!(v, weight_ref),
            bias, stride, padding, output_padding, groups, dilation
          )
        end

      # ======== Normalization ========
      t when t in ["torch.ops.aten.batch_norm.default",
                   "torch.ops.aten._native_batch_norm_legit_no_training.default"] ->
        input_ref = tensor_ref!(i, "input")
        weight_ref = tensor_ref_opt(i, "weight")
        bias_ref = tensor_ref_opt(i, "bias")
        rm_ref = tensor_ref_opt(i, "running_mean")
        rv_ref = tensor_ref_opt(i, "running_var")
        training = resolve_bool(i, "training", false)
        momentum = resolve_float(i, "momentum", 0.1)
        eps = resolve_float(i, "eps", 1.0e-5)
        fn v ->
          ExTorch.Native.aten_batch_norm(
            Map.fetch!(v, input_ref),
            if(weight_ref, do: Map.get(v, weight_ref), else: nil),
            if(bias_ref, do: Map.get(v, bias_ref), else: nil),
            if(rm_ref, do: Map.get(v, rm_ref), else: nil),
            if(rv_ref, do: Map.get(v, rv_ref), else: nil),
            training, momentum, eps
          )
        end

      t when t in ["torch.ops.aten.layer_norm.default",
                   "torch.ops.aten.native_layer_norm.default"] ->
        input_ref = tensor_ref_any(i, ["input", "self"])
        weight_ref = tensor_ref_opt(i, "weight")
        bias_ref = tensor_ref_opt(i, "bias")
        eps = resolve_float(i, "eps", 1.0e-5)
        normalized_shape = resolve_int_list_flex(i, "normalized_shape", [])
        fn v ->
          input = Map.fetch!(v, input_ref)
          shape = if normalized_shape == [],
            do: [elem(input.size, tuple_size(input.size) - 1)],
            else: normalized_shape
          ExTorch.Native.aten_layer_norm(
            input, shape,
            if(weight_ref, do: Map.get(v, weight_ref), else: nil),
            if(bias_ref, do: Map.get(v, bias_ref), else: nil),
            eps
          )
        end

      "torch.ops.aten.native_group_norm.default" ->
        input_ref = tensor_ref!(i, "input")
        weight_ref = tensor_ref_opt(i, "weight")
        bias_ref = tensor_ref_opt(i, "bias")
        num_groups = resolve_int(i, "group", resolve_int(i, "num_groups", 1))
        eps = resolve_float(i, "eps", 1.0e-5)
        fn v ->
          ExTorch.Native.aten_group_norm(
            Map.fetch!(v, input_ref),
            num_groups,
            if(weight_ref, do: Map.get(v, weight_ref), else: nil),
            if(bias_ref, do: Map.get(v, bias_ref), else: nil),
            eps
          )
        end

      # ======== Pooling ========
      t when t in ["torch.ops.aten.max_pool2d.default",
                   "torch.ops.aten.max_pool2d_with_indices.default"] ->
        self_ref = tensor_ref!(i, "self")
        kernel_size = resolve_int_list_flex(i, "kernel_size", [2, 2])
        stride = resolve_int_list_flex(i, "stride", kernel_size)
        padding = resolve_int_list_flex(i, "padding", [0, 0])
        dilation = resolve_int_list_flex(i, "dilation", [1, 1])
        ceil_mode = resolve_bool(i, "ceil_mode", false)
        fn v ->
          ExTorch.Native.aten_max_pool2d(
            Map.fetch!(v, self_ref),
            kernel_size, stride, padding, dilation, ceil_mode
          )
        end

      "torch.ops.aten.avg_pool2d.default" ->
        self_ref = tensor_ref!(i, "self")
        kernel_size = resolve_int_list_flex(i, "kernel_size", [2, 2])
        stride = resolve_int_list_flex(i, "stride", kernel_size)
        padding = resolve_int_list_flex(i, "padding", [0, 0])
        ceil_mode = resolve_bool(i, "ceil_mode", false)
        count_include_pad = resolve_bool(i, "count_include_pad", true)
        fn v ->
          ExTorch.Native.aten_avg_pool2d(
            Map.fetch!(v, self_ref),
            kernel_size, stride, padding, ceil_mode, count_include_pad
          )
        end

      t when t in ["torch.ops.aten.adaptive_avg_pool2d.default",
                   "torch.ops.aten._adaptive_avg_pool2d.default"] ->
        self_ref = tensor_ref!(i, "self")
        output_size = resolve_int_list_flex(i, "output_size", [1, 1])
        fn v ->
          ExTorch.Native.aten_adaptive_avg_pool2d(
            Map.fetch!(v, self_ref), output_size
          )
        end

      # ======== Activations ========
      t when t in ["torch.ops.aten.relu.default", "torch.ops.aten.relu_.default"] ->
        self_ref = tensor_ref!(i, "self")
        fn v -> ExTorch.functional_relu(Map.fetch!(v, self_ref)) end

      t when t in ["torch.ops.aten.hardtanh.default", "torch.ops.aten.hardtanh_.default"] ->
        self_ref = tensor_ref!(i, "self")
        min_val = resolve_float(i, "min_val", -1.0)
        max_val = resolve_float(i, "max_val", 1.0)
        fn v -> ExTorch.clamp(Map.fetch!(v, self_ref), min_val, max_val) end

      "torch.ops.aten.clamp.default" ->
        self_ref = tensor_ref!(i, "self")
        min_v = resolve_float(i, "min", -1.0e30)
        max_v = resolve_float(i, "max", 1.0e30)
        fn v -> ExTorch.clamp(Map.fetch!(v, self_ref), min_v, max_v) end

      # ======== Linear ========
      "torch.ops.aten.linear.default" ->
        input_ref = tensor_ref!(i, "input")
        weight_ref = tensor_ref!(i, "weight")
        bias_ref = tensor_ref_opt(i, "bias")
        fn v ->
          result = ExTorch.matmul(
            Map.fetch!(v, input_ref),
            ExTorch.transpose(Map.fetch!(v, weight_ref), 0, 1)
          )
          if bias_ref do
            case Map.get(v, bias_ref) do
              nil -> result
              bias -> ExTorch.add(result, bias)
            end
          else
            result
          end
        end

      # ======== Binary arithmetic (tensor/tensor) ========
      t when t in ["torch.ops.aten.add.Tensor", "torch.ops.aten.add_.Tensor"] ->
        compile_binary(i, &ExTorch.add/2)
      t when t in ["torch.ops.aten.mul.Tensor", "torch.ops.aten.mul_.Tensor"] ->
        compile_binary(i, &ExTorch.mul/2)
      "torch.ops.aten.sub.Tensor" ->
        compile_binary(i, &ExTorch.sub/2)
      "torch.ops.aten.div.Tensor" ->
        compile_binary(i, &ExTorch.tensor_div/2)

      # ======== Shape / passthrough ========
      "torch.ops.aten.flatten.using_ints" ->
        self_ref = tensor_ref!(i, "self")
        start_dim = resolve_int(i, "start_dim", 1)
        end_dim = resolve_int(i, "end_dim", -1)
        layer = ExTorch.NN.flatten(start_dim: start_dim, end_dim: end_dim)
        fn v -> ExTorch.NN.forward(Map.fetch!(v, self_ref), layer) end

      t when t in ["torch.ops.aten.native_dropout.default", "torch.ops.aten.dropout.default"] ->
        input_ref = tensor_ref!(i, "input")
        fn v -> Map.fetch!(v, input_ref) end

      # ======== Phase A: recurrent / transformer ========
      "torch.ops.aten.lstm.input" ->
        input_ref = tensor_ref!(i, "input")
        hx_refs = tensor_refs_list(i, "hx")
        params_refs = tensor_refs_list(i, "params")
        has_biases = resolve_bool(i, "has_biases", true)
        num_layers = resolve_int(i, "num_layers", 1)
        dropout = resolve_float(i, "dropout", 0.0)
        train = resolve_bool(i, "train", false)
        bidirectional = resolve_bool(i, "bidirectional", false)
        batch_first = resolve_bool(i, "batch_first", false)
        fn v ->
          hx = Enum.map(hx_refs, &Map.fetch!(v, &1))
          params = Enum.map(params_refs, &Map.fetch!(v, &1))
          ExTorch.Native.aten_lstm(
            Map.fetch!(v, input_ref), hx, params,
            has_biases, num_layers, dropout, train, bidirectional, batch_first
          )
        end

      "torch.ops.aten._transformer_encoder_layer_fwd.default" ->
        src_ref = tensor_ref!(i, "src")
        embed_dim = resolve_int(i, "embed_dim", 0)
        num_heads = resolve_int(i, "num_heads", 1)
        qkv_w = tensor_ref!(i, "qkv_weight")
        qkv_b = tensor_ref!(i, "qkv_bias")
        proj_w = tensor_ref!(i, "proj_weight")
        proj_b = tensor_ref!(i, "proj_bias")
        use_gelu = resolve_bool(i, "use_gelu", false)
        norm_first = resolve_bool(i, "norm_first", false)
        eps = resolve_float(i, "eps", 1.0e-5)
        n1w = tensor_ref!(i, "norm_weight_1")
        n1b = tensor_ref!(i, "norm_bias_1")
        n2w = tensor_ref!(i, "norm_weight_2")
        n2b = tensor_ref!(i, "norm_bias_2")
        ff1w = tensor_ref!(i, "ffn_weight_1")
        ff1b = tensor_ref!(i, "ffn_bias_1")
        ff2w = tensor_ref!(i, "ffn_weight_2")
        ff2b = tensor_ref!(i, "ffn_bias_2")
        fn v ->
          ExTorch.Native.aten_transformer_encoder_layer_fwd(
            Map.fetch!(v, src_ref), embed_dim, num_heads,
            Map.fetch!(v, qkv_w), Map.fetch!(v, qkv_b),
            Map.fetch!(v, proj_w), Map.fetch!(v, proj_b),
            use_gelu, norm_first, eps,
            Map.fetch!(v, n1w), Map.fetch!(v, n1b),
            Map.fetch!(v, n2w), Map.fetch!(v, n2b),
            Map.fetch!(v, ff1w), Map.fetch!(v, ff1b),
            Map.fetch!(v, ff2w), Map.fetch!(v, ff2b)
          )
        end

      # ======== Custom op handlers (ExTorch.Export.OpRegistry) ========
      target when is_binary(target) ->
        case ExTorch.Export.OpRegistry.lookup(target) do
          {:ok, handler} ->
            _ = initial_values
            handler.compile(node, initial_values, device)
          :error ->
            # Fall back to legacy dynamic dispatch for built-in ops
            # not yet in the compiled path.
            _ = initial_values
            fallback_runner(node, device)
        end
    end
  end

  # Binary Tensor-op compile helper. Handles (tensor, tensor) and
  # (tensor, scalar) by pre-deciding the branch at compile time.
  defp compile_binary(inputs, op) do
    self_ref = tensor_ref!(inputs, "self")
    case List.keyfind(inputs, "other", 0) do
      {"other", {:tensor, other_ref}} ->
        fn v -> op.(Map.fetch!(v, self_ref), Map.fetch!(v, other_ref)) end
      {"other", {:float, val}} ->
        fn v ->
          s = Map.fetch!(v, self_ref)
          op.(s, ExTorch.full(s.size, val, device: s.device))
        end
      {"other", {:int, val}} ->
        fn v ->
          s = Map.fetch!(v, self_ref)
          op.(s, ExTorch.full(s.size, val, device: s.device))
        end
      _ -> raise "Missing 'other' input for binary op"
    end
  end

  # Move all weight tensors to `device` in one pass. No-op when the target
  # is CPU (the default). Called from `load/2` with the :device option.
  defp maybe_move_weights(weights, :cpu), do: weights
  defp maybe_move_weights(weights, device) do
    for {fqn, tensor} <- weights, into: %{} do
      {fqn, ExTorch.Tensor.to(tensor, device: device)}
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

  defp execute_node(node, values, device) do
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
      "torch.ops.aten.rsqrt.default" ->
        s = resolve(i, "self", values)
        ExTorch.tensor_div(ExTorch.ones(s.size, device: s.device), ExTorch.tensor_sqrt(s))
      "torch.ops.aten.sin.default" -> ExTorch.tensor_sin(resolve(i, "self", values))
      "torch.ops.aten.cos.default" -> ExTorch.tensor_cos(resolve(i, "self", values))
      "torch.ops.aten.reciprocal.default" ->
        s = resolve(i, "self", values)
        ExTorch.tensor_div(ExTorch.ones(s.size, device: s.device), s)

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
      "torch.ops.aten.zeros.default" ->
        size = resolve_int_list_flex(i, "size", [1])
        ExTorch.zeros(List.to_tuple(size), device: device)
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
      "torch.ops.aten.conv_transpose2d.input" ->
        execute_conv_transpose2d(i, values)

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

      # ==== Pooling (direct at:: dispatch) ====
      "torch.ops.aten._adaptive_avg_pool2d.default" ->
        ExTorch.Native.aten_adaptive_avg_pool2d(
          resolve(i, "self", values),
          resolve_int_list_flex(i, "output_size", [1, 1]))
      "torch.ops.aten.adaptive_avg_pool2d.default" ->
        ExTorch.Native.aten_adaptive_avg_pool2d(
          resolve(i, "self", values),
          resolve_int_list_flex(i, "output_size", [1, 1]))
      "torch.ops.aten.max_pool2d.default" ->
        execute_max_pool2d(i, values)
      "torch.ops.aten.max_pool2d_with_indices.default" ->
        execute_max_pool2d(i, values)
      "torch.ops.aten.avg_pool2d.default" ->
        ExTorch.Native.aten_avg_pool2d(
          resolve(i, "self", values),
          resolve_int_list_flex(i, "kernel_size", [2, 2]),
          resolve_int_list_flex(i, "stride", resolve_int_list_flex(i, "kernel_size", [2, 2])),
          resolve_int_list_flex(i, "padding", [0, 0]),
          resolve_bool(i, "ceil_mode", false),
          resolve_bool(i, "count_include_pad", true))

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

      # ==== Recurrent (LSTM, GRU) ====
      "torch.ops.aten.lstm.input" ->
        execute_lstm(i, values)
      "torch.ops.aten.gru.input" ->
        execute_gru(i, values)

      # ==== Fused transformer encoder layer ====
      "torch.ops.aten._transformer_encoder_layer_fwd.default" ->
        execute_transformer_encoder_layer(i, values)

      # ==== Scaled dot-product attention ====
      "torch.ops.aten.scaled_dot_product_attention.default" ->
        execute_scaled_dot_product_attention(i, values)

      # ==== Native multi-head attention (legacy) ====
      "torch.ops.aten._native_multi_head_attention.default" ->
        execute_native_multi_head_attention(i, values)

      other ->
        # Check the custom op registry before giving up.
        case ExTorch.Export.OpRegistry.lookup(other) do
          {:ok, handler} ->
            runner = handler.compile(node, values, device)
            runner.(values)
          :error ->
            raise "Unsupported op: #{other}. Register a handler via ExTorch.Export.OpRegistry " <>
                  "or use ExTorch.AOTI for compiled inference."
        end
    end
  end

  # Complex op helpers — Phase B: direct at:: dispatch via NIFs.
  # The previous versions allocated a torch::nn::Module, copied parameters
  # in, ran forward, then discarded the layer on every call. The new versions
  # call at::conv2d / at::batch_norm / etc. functionally — no per-call layer
  # construction, no parameter copy, ~5x faster on deep CNNs.

  defp execute_conv2d(i, values) do
    ExTorch.Native.aten_conv2d(
      resolve(i, "input", values),
      resolve(i, "weight", values),
      resolve_optional(i, "bias", values),
      resolve_int_list_flex(i, "stride", [1, 1]),
      resolve_int_list_flex(i, "padding", [0, 0]),
      resolve_int_list_flex(i, "dilation", [1, 1]),
      resolve_int(i, "groups", 1)
    )
  end

  defp execute_convolution(i, values) do
    weight = resolve(i, "weight", values)
    ndim = tuple_size(weight.size) - 2
    bias = resolve_optional(i, "bias", values)
    input = resolve(i, "input", values)
    stride = resolve_int_list_flex(i, "stride", [1])
    padding = resolve_int_list_flex(i, "padding", [0])
    dilation = resolve_int_list_flex(i, "dilation", [1])
    groups = resolve_int(i, "groups", 1)
    transposed = resolve_bool(i, "transposed", false)
    output_padding = resolve_int_list_flex(i, "output_padding", List.duplicate(0, ndim))

    if transposed do
      ExTorch.Native.aten_convolution(
        input, weight, bias, stride, padding, dilation, true, output_padding, groups
      )
    else
      case ndim do
        1 -> ExTorch.Native.aten_conv1d(input, weight, bias, stride, padding, dilation, groups)
        2 -> ExTorch.Native.aten_conv2d(input, weight, bias, stride, padding, dilation, groups)
        3 -> ExTorch.Native.aten_conv3d(input, weight, bias, stride, padding, dilation, groups)
      end
    end
  end

  defp execute_conv_transpose2d(i, values) do
    ExTorch.Native.aten_conv_transpose2d(
      resolve(i, "input", values),
      resolve(i, "weight", values),
      resolve_optional(i, "bias", values),
      resolve_int_list_flex(i, "stride", [1, 1]),
      resolve_int_list_flex(i, "padding", [0, 0]),
      resolve_int_list_flex(i, "output_padding", [0, 0]),
      resolve_int(i, "groups", 1),
      resolve_int_list_flex(i, "dilation", [1, 1])
    )
  end

  defp execute_max_pool2d(i, values) do
    ExTorch.Native.aten_max_pool2d(
      resolve(i, "self", values),
      resolve_int_list_flex(i, "kernel_size", [2, 2]),
      resolve_int_list_flex(i, "stride", resolve_int_list_flex(i, "kernel_size", [2, 2])),
      resolve_int_list_flex(i, "padding", [0, 0]),
      resolve_int_list_flex(i, "dilation", [1, 1]),
      resolve_bool(i, "ceil_mode", false)
    )
  end

  defp execute_batch_norm(i, values) do
    ExTorch.Native.aten_batch_norm(
      resolve(i, "input", values),
      resolve_optional(i, "weight", values),
      resolve_optional(i, "bias", values),
      resolve_optional(i, "running_mean", values),
      resolve_optional(i, "running_var", values),
      resolve_bool(i, "training", false),
      resolve_float(i, "momentum", 0.1),
      resolve_float(i, "eps", 1.0e-5)
    )
  end

  defp execute_layer_norm(i, values) do
    input = resolve_any(i, ["input", "self"], values)
    normalized_shape = resolve_int_list_flex(i, "normalized_shape",
      [elem(input.size, tuple_size(input.size) - 1)])
    ExTorch.Native.aten_layer_norm(
      input,
      normalized_shape,
      resolve_optional(i, "weight", values),
      resolve_optional(i, "bias", values),
      resolve_float(i, "eps", 1.0e-5)
    )
  end

  defp execute_group_norm(i, values) do
    ExTorch.Native.aten_group_norm(
      resolve(i, "input", values),
      resolve_int(i, "group", resolve_int(i, "num_groups", 1)),
      resolve_optional(i, "weight", values),
      resolve_optional(i, "bias", values),
      resolve_float(i, "eps", 1.0e-5)
    )
  end

  # Direct dispatch to at::lstm via NIF. Replaces the previous Elixir-side
  # cell loop, which paid one ExTorch call per gate per timestep. The exported
  # aten op returns (output, h_n, c_n); we return them as a list so the main
  # forward loop binds them to the node's three output names.
  defp execute_lstm(i, values) do
    input = resolve(i, "input", values)
    hx = resolve_tensor_list(i, "hx", values)
    params = resolve_tensor_list(i, "params", values)
    has_biases = resolve_bool(i, "has_biases", true)
    num_layers = resolve_int(i, "num_layers", 1)
    dropout = resolve_float(i, "dropout", 0.0)
    train = resolve_bool(i, "train", false)
    bidirectional = resolve_bool(i, "bidirectional", false)
    batch_first = resolve_bool(i, "batch_first", false)

    ExTorch.Native.aten_lstm(
      input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first
    )
  end

  defp execute_gru(i, values) do
    input = resolve(i, "input", values)
    hx = resolve(i, "hx", values)
    params = resolve_tensor_list(i, "params", values)
    has_biases = resolve_bool(i, "has_biases", true)
    num_layers = resolve_int(i, "num_layers", 1)
    dropout = resolve_float(i, "dropout", 0.0)
    train = resolve_bool(i, "train", false)
    bidirectional = resolve_bool(i, "bidirectional", false)
    batch_first = resolve_bool(i, "batch_first", false)

    ExTorch.Native.aten_gru(
      input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first
    )
  end

  # Direct dispatch to at::_transformer_encoder_layer_fwd. Replaces the
  # previous Elixir-side unroll, which spent ~30 NIF calls per layer
  # constructing intermediate tensors.
  defp execute_transformer_encoder_layer(i, values) do
    ExTorch.Native.aten_transformer_encoder_layer_fwd(
      resolve(i, "src", values),
      resolve_int(i, "embed_dim", 0),
      resolve_int(i, "num_heads", 1),
      resolve(i, "qkv_weight", values),
      resolve(i, "qkv_bias", values),
      resolve(i, "proj_weight", values),
      resolve(i, "proj_bias", values),
      resolve_bool(i, "use_gelu", false),
      resolve_bool(i, "norm_first", false),
      resolve_float(i, "eps", 1.0e-5),
      resolve(i, "norm_weight_1", values),
      resolve(i, "norm_bias_1", values),
      resolve(i, "norm_weight_2", values),
      resolve(i, "norm_bias_2", values),
      resolve(i, "ffn_weight_1", values),
      resolve(i, "ffn_bias_1", values),
      resolve(i, "ffn_weight_2", values),
      resolve(i, "ffn_bias_2", values)
    )
  end

  defp execute_scaled_dot_product_attention(i, values) do
    {scale, has_scale} = case List.keyfind(i, "scale", 0) do
      {"scale", {:float, v}} -> {v, true}
      {"scale", {:int, v}} -> {v / 1, true}
      _ -> {0.0, false}
    end

    ExTorch.Native.aten_scaled_dot_product_attention(
      resolve(i, "query", values),
      resolve(i, "key", values),
      resolve(i, "value", values),
      resolve_float(i, "dropout_p", 0.0),
      resolve_bool(i, "is_causal", false),
      scale,
      has_scale
    )
  end

  defp execute_native_multi_head_attention(i, values) do
    ExTorch.Native.aten_native_multi_head_attention(
      resolve(i, "query", values),
      resolve(i, "key", values),
      resolve(i, "value", values),
      resolve_int(i, "embed_dim", 0),
      resolve_int(i, "num_head", 1),
      resolve(i, "qkv_weight", values),
      resolve(i, "qkv_bias", values),
      resolve(i, "proj_weight", values),
      resolve(i, "proj_bias", values),
      resolve_bool(i, "need_weights", false),
      resolve_bool(i, "average_attn_weights", true)
    )
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
      {^name, {:float, val}} -> ExTorch.full(ref_tensor.size, val, device: ref_tensor.device)
      {^name, {:int, val}} -> ExTorch.full(ref_tensor.size, val, device: ref_tensor.device)
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

  defp resolve_bool(inputs, name, default) do
    case List.keyfind(inputs, name, 0) do
      {^name, {:bool, val}} -> val
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

    # Count how many times each value is referenced as an input downstream
    ref_counts = count_references(graph)
    last_index = length(graph) - 1

    # Walk nodes and generate code lines
    {lines, _} =
      graph
      |> Enum.with_index()
      |> Enum.reduce({[], %{user_input => "x"}}, fn {node, idx}, {lines_acc, var_map} ->
        output_name = List.first(node.outputs) || "unknown"
        expr = node_to_elixir(node, var_map, schema.weights)
        var_name = sanitize_var(output_name)
        used_count = Map.get(ref_counts, output_name, 0)
        is_last = idx == last_index

        # - Final node with no downstream use: emit a bare expression so it
        #   becomes the function's return value with no dead binding.
        # - Intermediate node with no downstream use (rare; effectively dead
        #   code in the graph): keep the binding but underscore-prefix it to
        #   suppress the unused-variable warning.
        # - Otherwise: regular binding so branching consumers (e.g. skip
        #   connections) can refer back to it.
        line =
          cond do
            is_last and used_count == 0 -> "    #{expr}"
            used_count == 0 -> "    _#{var_name} = #{expr}"
            true -> "    #{var_name} = #{expr}"
          end

        {lines_acc ++ [line], Map.put(var_map, output_name, var_name)}
      end)

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
