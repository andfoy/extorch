defmodule ExTorch.Export.OpHandler do
  @moduledoc """
  Behaviour for extending `ExTorch.Export` with custom op implementations.

  Implement this behaviour to add support for ops that aren't built into
  the ExTorch.Export interpreter (e.g., torchvision ops like `roi_align`,
  `nms`, or domain-specific custom ops).

  ## Usage

  1. Implement the behaviour:

      defmodule MyApp.VisionOps do
        @behaviour ExTorch.Export.OpHandler

        @impl true
        def ops do
          ["torchvision::roi_align", "torchvision::nms"]
        end

        @impl true
        def compile(node, _initial_values, device) do
          case node.target do
            "torchvision::roi_align" ->
              # Return a closure that will be called during forward/2
              fn values -> my_roi_align(node.inputs, values, device) end
            "torchvision::nms" ->
              fn values -> my_nms(node.inputs, values, device) end
          end
        end
      end

  2. Register the handler in your application config:

      config :extorch, :export_op_handlers, [MyApp.VisionOps]

  Or register at runtime:

      ExTorch.Export.OpRegistry.register(MyApp.VisionOps)

  ## Callbacks

  - `ops/0` — Returns the list of op target strings this handler supports.
    Used to build the dispatch index at registration time.

  - `compile/3` — Called at `ExTorch.Export.load/2` time for each matched node.
    Returns a closure `(values_map -> tensor | [tensor])` that will be invoked
    during `forward/2`. The closure should capture any scalar args or tensor
    refs from `node.inputs` so forward is a tight loop.
  """

  @typedoc "A graph node from the exported model schema."
  @type graph_node :: %{target: String.t(), inputs: list(), outputs: list()}

  @typedoc "The device the model was loaded onto (`:cpu`, `:cuda`, `{:cuda, 0}`, etc.)."
  @type device :: atom() | {atom(), non_neg_integer()}

  @doc """
  Returns the list of op target strings this handler supports.

  Example: `["torchvision::roi_align", "torchvision::nms"]`
  """
  @callback ops() :: [String.t()]

  @doc """
  Compile a graph node into a forward-time closure.

  Called once per matched node at `ExTorch.Export.load/2` time.
  `initial_values` contains the pre-resolved parameter/buffer tensors.
  `device` is the target device passed to `load/2`.

  Must return a function `(values_map -> tensor | [tensor])` where
  `values_map` is the running map of node output name → tensor.
  """
  @callback compile(node :: graph_node(), initial_values :: map(), device :: device()) ::
              (map() -> ExTorch.Tensor.t() | [ExTorch.Tensor.t()])
end
