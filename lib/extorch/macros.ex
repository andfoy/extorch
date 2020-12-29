defmodule ExTorch.Macros do
  @moduledoc false

  defp filter_arg({:\\, _, [arg, {:"::", _, [default, specs]}]}) do
    {{:\\, [], [arg, default]}, specs}
  end

  defp filter_arg({:"::", _, [arg, specs]}) do
    {arg, specs}
  end

  defp remove_args_specs(args) do
    args
    |> Enum.map(&filter_arg/1)
    |> Enum.unzip()
  end

  defp get_doc_map() do
    case Code.fetch_docs(ExTorch.Native) do
      {:docs_v1, _, :elixir, "text/markdown", _, _, funcs} ->
        # :logger.debug("Funcs: #{inspect(funcs)}")
        funcs
        |> Enum.reduce([], fn
          {{:function, fn_name, _}, _, _, doc_info, _}, acc ->
            case doc_info do
              %{"en" => doc_string} -> [{fn_name, doc_string} | acc]
              _ -> acc
            end

          x, acc ->
            :logger.debug(x)
            acc
        end)
        |> Enum.into(%{})

      _ ->
        %{}
    end
  end

  defp extract_arg({:"::", _, [arg | _]}) do
    arg
  end

  defp extract_arg({:\\, _, [arg | _]}) do
    arg
  end

  defp extract_arg(arg) do
    arg
  end

  defp get_args(args) do
    Enum.map(args, &extract_arg/1)
  end

  defp kwarg_map({:\\, _, [{arg, _, _}, value]}) do
    %{arg => value}
  end

  defp body(name, args, extended_body, return_spec) when length(args) > 0 do
    quote do
      @spec unquote(name)(any(), [{:key, integer()} | {:key2, integer()}]) :: ExTorch.Tensor.t()
      def unquote(name)(unquote_splicing(args), unquote(Macro.var(:kwargs, nil))) do
        unquote(extended_body)
      end

      @doc false
      # @spec unquote(name)(any()) :: ExTorch.Tensor.t()
      def unquote(name)(unquote_splicing(args)) do
        unquote(Macro.var(:kwargs, nil)) = %{}
        unquote(extended_body)
      end
    end
  end

  defp body(name, args, extended_body, return_spec) when length(args) == 0 do
    quote do
      def unquote(name)(unquote(Macro.var(:kwargs, nil))) do
        unquote(extended_body)
      end

      @doc false
      def unquote(name)() do
        unquote(Macro.var(:kwargs, nil)) = %{}
        unquote(extended_body)
      end
    end
  end

  # kwargs = [:kwarg1 : value, :kwarg2: value2, ...]
  # def name(arg1, arg2, ..., kwargs) do
  # Join __CALLER__ args with function signature
  # Unpack keyword list into variables
  # Body
  # end

  defmacro deftensor(fn_name_spec, do: fn_body) do
    docs = get_doc_map()
    {:"::", _, [fn_name_args, return_spec]} = fn_name_spec
    {name, _, args} = fn_name_args

    {args, args_spec} = remove_args_specs(args)
    :logger.debug("#{inspect args}")
    :logger.debug("#{inspect args_spec}")
    kwargs = Enum.filter(args, fn e -> match?({:\\, _, _}, e) end)
    args = Enum.filter(args, fn e -> !match?({:\\, _, _}, e) end)
    # :logger.debug("#{inspect args}, #{inspect kwargs}")
    kwargs = Enum.map(kwargs, &kwarg_map/1)
    fn_kwargs = Enum.reduce(kwargs, %{}, fn x, acc -> Map.merge(x, acc) end)
    :logger.debug("#{inspect fn_kwargs}")

    fn_kwargs_var = Macro.escape(fn_kwargs)
    args_reassignment = Enum.map(fn_kwargs, fn {k, _} -> {k, Macro.var(k, nil)} end)
    kwargs_assignment = {:%{}, [], args_reassignment}

    new_body =
      quote do
        unquote(Macro.var(:kwargs, nil)) =
          Enum.into(
            unquote(Macro.var(:kwargs, nil)),
            %{}
          )

        unquote(Macro.var(:kwargs, nil)) =
          Map.merge(
            unquote(fn_kwargs_var),
            unquote(Macro.var(:kwargs, nil))
          )

        unquote(kwargs_assignment) = unquote(Macro.var(:kwargs, nil))
        unquote(fn_body)
        # unquote(Macro.var(:tensor_result, nil)) = unquote(fn_body)
        # ExTorch.Tensor.wrap_tensor_ref(unquote(Macro.var(:tensor_result, nil)))
      end

    actual_body = body(name, args, new_body, return_spec)
    :logger.debug("#{Macro.to_string(Macro.expand(actual_body, nil))}")
    actual_body
  end

  defmacro native_calls(do: body) do
    {:__block__, _, native_defs} = body

    native_module = {:__aliases__, [alias: false], [:ExTorch, :Native]}
    redefinitions =
      Enum.map(native_defs, fn {:"::", _, [fun_def = {fun_name, _, fun_args}, ret_spec]} ->
        all_args = get_args(fun_args)
        native_call = {:., [], [native_module, fun_name]}
        quote do
          deftensor unquote(fun_def) :: unquote(ret_spec) do
            unquote(native_call)(unquote_splicing(all_args))
          end
        end

        fun_def = {fun_name, _, fun_args} ->
          all_args = get_args(fun_args)
          native_call = {:., [], [native_module, fun_name]}
          quote do
            deftensor unquote(fun_def) :: nil do
              unquote(native_call)(unquote_splicing(all_args))
            end
          end
      end)

    {:__block__, [], redefinitions}
  end
end
