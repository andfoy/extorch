defmodule ExTorch.ModuleMixin do
  @signature_regex ~r/[a-zA-Z_]\w*[(]((\w,? ?)*)[)]/

  defmacro extends(module) do
    module = Macro.expand(module, __CALLER__)
    functions = module.__info__(:functions)

    {signature_args, doc_funcs} =
      module
      |> Code.fetch_docs()
      |> fetch_signatures()

    agg_functions =
      Enum.reduce(functions, %{}, fn {func, arity}, acc ->
        current_max = Map.get(acc, func, 0)
        Map.put(acc, func, max(current_max, arity))
      end)

    signatures =
      Enum.reduce(functions, [], fn {name, arity}, acc ->
        str_name = Atom.to_string(name)

        case String.starts_with?(str_name, "__") do
          false ->
            docstring = Map.get(doc_funcs, {name, arity})

            add_max_header =
              case docstring do
                nil -> false
                docstring -> String.contains?(docstring, "Available signature calls")
              end

            signature = Map.get(signature_args, {name, arity})
            [_, args, _] = Regex.run(@signature_regex, signature)
            args = String.split(args, ",", trim: true)
            args = Enum.map(args, fn arg -> Macro.var(String.to_atom(arg), nil) end)

            [{{name, [], args}, add_max_header} | acc]

          true ->
            acc
        end
      end)

    # zipped = List.zip([signatures, functions])
    Enum.map(signatures, fn
      {sig, true} ->
        {func_name, _, _} = sig
        max_arity = Map.get(agg_functions, func_name)

        quote do
          ExTorch.DelegateWithDocs.defdelegate(unquote(sig),
            to: unquote(module),
            point_to: unquote(max_arity)
          )
        end

      {sig, false} ->
        quote do
          ExTorch.DelegateWithDocs.defdelegate(unquote(sig), to: unquote(module))
        end
    end)
  end

  defp fetch_signatures({:docs_v1, _, :elixir, "text/markdown", _, %{}, funcs}) do
    Enum.reduce(funcs, {%{}, %{}}, fn
      {{:function, func_name, arity}, _, [signature], docstring, _}, {acc, docs} ->
        docstring =
          case docstring do
            %{"en" => doc} -> doc
            _ -> nil
          end

        acc = Map.put(acc, {func_name, arity}, signature)
        docs = Map.put(docs, {func_name, arity}, docstring)
        {acc, docs}

      _, acc ->
        acc
    end)
  end
end
