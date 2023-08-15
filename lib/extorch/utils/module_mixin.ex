defmodule ExTorch.ModuleMixin do
  @moduledoc """
  Utilities used to define a module mixin that inherits documentation and specs.
  """

  @doc """
  This macro enables a module to import the functions from another module
  and expose them as they were defined on it.

      defmodule BaseModule do
        def call1(arg1, arg2) do
          arg1 + arg2
        end

        def call2() do
          :ok
        end
      end

      defmodule Mixin do
        import ExTorch.ModuleMixin
        extends(BaseModule)
      end

  By using the `extends/1` macro, the `Mixin` module will have the definitions
  of `call1/2` and `call2/0`.

  ## Implementation notes
  The function definitions are given via `defdelegate` internally.
  """
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
            add_max_header = is_max_arity_call(doc_funcs, name, arity)
            args = get_arguments(signature_args, name, arity)
            discard_optional_signatures(name, args, add_max_header, acc)

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

  defp discard_optional_signatures(name, args, add_max_header, acc) do
    case args do
      nil ->
        acc

      _ ->
        [{{name, [], args}, add_max_header} | acc]
    end
  end

  defp get_arguments(signature_args, name, arity) do
    signature = Map.get(signature_args, {name, arity})

    case signature do
      nil ->
        nil

      _ ->
        case Code.string_to_quoted(signature) do
          {:ok, {_, _, args}} ->
            args

          _ ->
            nil
        end
    end
  end

  defp is_max_arity_call(doc_funcs, name, arity) do
    docstring = Map.get(doc_funcs, {name, arity})

    case docstring do
      nil -> false
      docstring -> String.contains?(docstring, "Available signature calls")
    end
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
