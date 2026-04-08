defmodule ExTorch.NN.JITBackedModel do
  @moduledoc """
  A model instance backed by a loaded TorchScript (.pt) file.

  This struct is returned by `MyModule.from_jit/1` and holds the underlying
  JIT model. When used with the DSL's `layer/3` function, forward passes
  are delegated to the JIT model's `forward` method directly, using the
  pre-trained weights from the `.pt` file.

  The DSL module definition serves as a structural contract -- the declared
  layers are validated against the JIT model's submodules at load time.
  """

  @type t :: %__MODULE__{
          jit_model: ExTorch.JIT.Model.t(),
          module_name: atom(),
          layer_names: [atom()]
        }

  defstruct [:jit_model, :module_name, :layer_names]
end
