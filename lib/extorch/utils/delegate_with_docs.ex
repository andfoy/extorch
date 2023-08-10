defmodule ExTorch.DelegateWithDocs do
  @moduledoc """
  Public API documentation for `DelegateWithDocs`. This module is based on
  https://github.com/danielberkompas/delegate_with_docs
  """

  import Kernel, except: [defdelegate: 2]
  require Kernel

  defmodule Error do
    defexception [:message]
  end

  @doc """
  Overrides `Kernel.defdelegate/2` with `defdelegate/2`.
  """
  defmacro __using__(_) do
    quote do
      import Kernel, except: [defdelegate: 2]
      require Kernel

      import ExTorch.DelegateWithDocs

      # Hack to ensure that the module's docs and specs
      # are available to other modules at compile time.
      #
      # For some reason, Elixir waits until all the modules
      # are compiled before writing anything to disk, which
      # means that docs are not available on the first compile.
      #
      # We can circumvent this by writing a module's bytecode
      # to the proper location after it is compiled.
      @after_compile __MODULE__
      def __after_compile__(_env, bytecode) do
        __MODULE__
        |> :code.which()
        |> to_string()
        |> File.write(bytecode)
      end
    end
  end

  @doc """
  Delegates a function to another module, copying its docs.
  Use exactly like `Kernel.defdelegate/2`.
  """
  defmacro defdelegate(fun, opts) do
    # caller_module =
    caller_module =
      __CALLER__.module
      |> Atom.to_string()
      |> String.replace("Elixir.", "")

    {options, _} = Code.eval_quoted(opts, [], __CALLER__)
    {function_alias, _, args} = fun

    module = options[:to]
    function = options[:as] || function_alias
    point_to_docs = options[:point_to]
    signature = {function, length(args)}

    {doc_spec, specs} =
      case get_doc(module, signature) do
        {doc_str, doc_attrs} ->
          doc_spec =
            assemble_doc_specs(function, caller_module, point_to_docs, doc_str, doc_attrs)

          specs = get_specs(module, signature)
          {doc_spec, specs}

        nil ->
          {nil, nil}
      end

    delegate =
      quote do
        unquote(doc_spec)
        unquote(specs)
        Kernel.defdelegate(unquote(fun), unquote(opts))
      end

    delegate
  end

  defp assemble_doc_specs(function, caller_module, point_to_docs, doc_str, doc_attrs) do
    doc_specs =
      Enum.reduce(doc_attrs, [], fn
        {:defaults, _}, acc ->
          acc

        {k, v}, acc ->
          doc_part =
            quote do
              @doc [{unquote(k), unquote(v)}]
            end

          [doc_part] ++ acc
      end)

    doc_specs =
      case doc_str do
        nil ->
          doc_specs

        _ ->
          doc_str =
            case point_to_docs do
              nil ->
                doc_str

              _ ->
                """
                See `#{caller_module}.#{Atom.to_string(function)}/#{point_to_docs}`

                #{doc_str}
                """
            end

          doc_str_quote =
            quote do
              @doc unquote(doc_str)
            end

          [doc_str_quote | doc_specs]
      end

    {:__block__, [], doc_specs}
  end

  @doc """
  Get the doc string for a given module and function.
  ## Example
      DelegateWithDocs.get_doc(MyModule.Internal, {:my_func, 2})
  """
  @spec get_doc(module, {atom, integer}) :: {String.t() | nil, map() | :none} | nil
  def get_doc(module, {function, arity}) do
    assert_module_exists!(module)

    module
    |> Code.fetch_docs()
    |> fetch_docs()
    |> Map.get({function, arity})
  end

  defp fetch_docs({:docs_v1, _, :elixir, "text/markdown", _, %{}, funcs}) do
    Enum.reduce(funcs, %{}, fn
      {{:function, func_name, arity}, _, _, docstring, sections}, acc ->
        docstring =
          case docstring do
            %{"en" => doc} -> doc
            _ -> nil
          end

        Map.put(acc, {func_name, arity}, {docstring, sections})

      _, acc ->
        acc
    end)
  end

  @doc """
  Get the typespecs for a given function as an AST.
  """
  def get_specs(module, {function, arity}) do
    assert_module_exists!(module)

    {:ok, module_specs} =
      module
      |> Code.Typespec.fetch_specs()

    module_specs = Enum.into(module_specs, %{})
    func_spec = Map.get(module_specs, {function, arity})

    quoted_specs =
      Enum.map(func_spec, fn spec ->
        quoted_spec = Code.Typespec.spec_to_quoted(function, spec)

        quote do
          @spec unquote(quoted_spec)
        end
      end)

    {:__block__, [], quoted_specs}
  end

  defp assert_module_exists!(module) do
    unless Code.ensure_compiled(module),
      do: raise(Error, "Module #{inspect(module)} is not defined/available")
  end
end
