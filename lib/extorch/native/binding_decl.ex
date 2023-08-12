defmodule ExTorch.Native.BindingDeclaration do
  @moduledoc """
  Conveniences for declaring native calls to a library in Rustler.

  This module can be `use`-d into a module in order to declare
  a set of native functions:

      defmodule NativeCalls do
        use ExTorch.Native.BindingDeclaration

        defbindings(:doc_section) do
          @doc \"\"\"
          Get the size of a tensor.

          ## Arguments
            - `tensor`: Input tensor
          \"\"\"
          @spec size(ExTorch.Tensor.t()) :: tuple()
          defbinding(size(tensor))
        end
      end

  ## Binding behaviour
  Internally, `ExTorch.Native.BindingDeclaration` implements the following
  macros:
  * `__using__/0`: Enables the module to be used on a module
  that uses `Rustler` in order to define the stub definitions for the native
  functions to call.


        defmodule NativeExtension do
          use NativeCalls
          use Rustler, otp_app: :app_name, crate: "crate_name"
        end
  """

  defmacro __using__(_opts) do
    caller_module = __CALLER__.module

    output_stub_decl =
      quote do
        args =
          if arity == 0 do
            []
          else
            Enum.map(1..arity, fn i ->
              {String.to_atom("_#{<<?x, ?A + i - 1>>}"), [], nil}
            end)
          end

        quote do
          @doc false
          def unquote(name)(unquote_splicing(args)) do
            :erlang.nif_error(:nif_not_loaded)
          end
        end
      end

    using_macro =
      quote do
        defmacro __using__(_opts) do
          inherit_funcs = unquote(caller_module).__info__(:functions)

          Enum.map(inherit_funcs, fn {name, arity} ->
            str_name = Atom.to_string(name)

            case String.starts_with?(str_name, "__") do
              false ->
                unquote(output_stub_decl)

              true ->
                {:__block__, [], []}
            end
          end)
        end
      end

    quote do
      Module.register_attribute(__MODULE__, :doc_section, persist: true)
      Module.register_attribute(__MODULE__, :nif_module, persist: true)
      use ExTorch.DelegateWithDocs

      import ExTorch.DType
      import ExTorch.Device
      import ExTorch.Layout
      import ExTorch.MemoryFormat
      import ExTorch.Native.Macros

      unquote(using_macro)
    end
  end

  defmacro __before_compile__(env) do
    doc_section = Module.get_attribute(env.module, :doc_section)

    quote do
      def doc_section do
        unquote(doc_section)
      end
    end
  end
end
