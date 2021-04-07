defmodule ExTorch.DelegateWithDocs do
  @moduledoc """
  Public API documentation for `DelegateWithDocs`. This module is based on
  https://github.com/danielberkompas/delegate_with_docs
  """

  import Kernel, except: [defdelegate: 2]
  require Kernel

  defmodule Error do
    @moduledoc false
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

    {doc_str, doc_attrs} = get_doc(module, signature)
    specs = get_specs(module, signature, function_alias)

    doc_spec = assemble_doc_specs(function, caller_module, point_to_docs, doc_str, doc_attrs)

    delegate = quote do
      unquote(doc_spec)
      unquote(specs)
      Kernel.defdelegate(unquote(fun), unquote(opts))
    end

    delegate
  end

  defp assemble_doc_specs(function, caller_module, point_to_docs, doc_str, doc_attrs) do
    doc_specs = Enum.map(doc_attrs, fn {k, v} ->
      quote do
        @doc [{unquote(k), unquote(v)}]
      end
    end)

    doc_specs =
      case doc_str do
        nil -> doc_specs
        _ ->
          doc_str =
            case point_to_docs do
              nil -> doc_str
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
  @spec get_doc(module, {atom, integer}) :: {String.t() | nil, map() | :none}
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
  def get_specs(module, {function, arity}, function_alias \\ nil) do
    function_alias = function_alias || function
    assert_module_exists!(module)

    {:ok, module_specs} =
      module
      |> Code.Typespec.fetch_specs()

    module_specs = Enum.into(module_specs, %{})
    func_spec = Map.get(module_specs, {function, arity})

    quoted_specs = Enum.map(func_spec, fn spec ->
      quoted_spec = Code.Typespec.spec_to_quoted(function, spec)
      quote do
         @spec unquote(quoted_spec)
      end
    end)
    {:__block__, [], quoted_specs}
  end

  # Line numbers must be recursively stripped out of the spec AST
  # to prevent errors when we inject the spec into the delegating
  # module
  defp remove_line_numbers(ast, acc) when ast in [[], nil], do: Enum.reverse(acc)

  defp remove_line_numbers([ast | tail], acc) do
    remove_line_numbers(tail, [remove_line_numbers(ast) | acc])
  end

  defp remove_line_numbers({ast, context, args}) when is_tuple(ast) do
    {remove_line_numbers(ast), Keyword.drop(context, [:line]), remove_line_numbers(args, [])}
  end

  defp remove_line_numbers({func, context, args}) when is_list(context) do
    {func, Keyword.drop(context, [:line]), remove_line_numbers(args, [])}
  end

  defp remove_line_numbers({func, line, args}) when is_integer(line) do
    {func, [], remove_line_numbers(args, [])}
  end

  defp remove_line_numbers(other), do: other

  # Renames the spec to the alias
  defp rename_function({:"::", sc, [{_function, fc, fargs}, return]}, function_alias) do
    {:"::", sc, [{function_alias, fc, fargs}, return]}
  end

  defp rename_function(ast, _as), do: ast

  defp assert_module_exists!(module) do
    unless Code.ensure_compiled(module),
      do: raise(Error, "Module #{inspect(module)} is not defined/available")

    # unless Code.get_docs(module, :docs) do
    #   raise(Error, """
    #   Module #{inspect(module)} was not compiled with docs.
    #   You must `use DelegateWithDocs` within #{inspect(module)} to ensure
    #   that its docs are available to other modules at compile time.
    #   """)
    # end
  end
end
