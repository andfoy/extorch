defmodule ExTorch.Native.BindingDeclaration do
  defmacro __using__(_opts) do
    caller_module = __CALLER__.module

    quote do
      Module.register_attribute(__MODULE__, :doc_section, persist: true)
      Module.register_attribute(__MODULE__, :nif_module, persist: true)
      use ExTorch.DelegateWithDocs

      import ExTorch.DType
      import ExTorch.Device
      import ExTorch.Layout
      import ExTorch.MemoryFormat
      import ExTorch.Native.Macros

      defmacro __using__(_opts) do
        inherit_funcs = unquote(caller_module).__info__(:functions)

        Enum.map(inherit_funcs, fn {name, arity} ->
          str_name = Atom.to_string(name)

          case String.starts_with?(str_name, "__") do
            false ->
              args =
                if arity == 0 do
                  []
                else
                  Enum.map(1..arity, fn i ->
                    {String.to_atom("_#{<<?x, ?A + i - 1>>}"), [], nil}
                  end)
                end

              quote do
                def unquote(name)(unquote_splicing(args)) do
                  :erlang.nif_error(:nif_not_loaded)
                end
              end

            true ->
              {:__block__, [], []}
          end
        end)
      end
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
