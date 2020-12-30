defmodule ExTorch.Macros do
  @moduledoc false

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

  defp get_specs() do
    case Code.Typespec.fetch_specs(ExTorch.Native) do
      {:ok, specs} ->
        specs
        |> Enum.map(fn {{name, _}, [spec]} ->
          quoted_spec = Code.Typespec.spec_to_quoted(name, spec)
          {name, quoted_spec}
        end)
        |> Enum.into(%{})

      :error ->
        %{}
    end
  end

  defp consume_args_spec([], kwargs_spec, acc) do
    {Enum.reverse(acc), kwargs_spec}
  end

  defp consume_args_spec([_arg|rest], [spec|args], acc) do
    consume_args_spec(rest, args, [spec|acc])
  end

  defp get_new_specs(_args, _kwargs, nil) do
    {[], []}
  end

  defp get_new_specs(args, kwargs, spec) do
    {:"::", _, [{name, _, sig_spec}, return_spec]} = spec
    {args_spec, kwargs_spec} = consume_args_spec(args, sig_spec, [])
    keyword_list_kwargs = Enum.zip([kwargs, kwargs_spec])
    new_sig_spec = args_spec ++ [keyword_list_kwargs]
    kwarg_spec = {:"::", [], [{name, [], new_sig_spec}, return_spec]}
    single_spec = {:"::", [], [{name, [], args_spec}, return_spec]}
    kwarg_spec = quote do
      @spec unquote(kwarg_spec)
    end
    single_spec = quote do
      @spec unquote(single_spec)
    end
    {single_spec, kwarg_spec}
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
    {arg, value}
  end

  defp body(name, args, [], extended_body, single_spec, kwarg_spec, doc) when length(args) > 0 do
    quote do
      # @spec unquote(name)(any(), [{:key, integer()} | {:key2, integer()}]) :: ExTorch.Tensor.t()
      @doc unquote(doc)
      unquote(single_spec)
      def unquote(name)(unquote_splicing(args)) do
        unquote(Macro.var(:kwargs, nil)) = %{}
        unquote(extended_body)
      end
    end
  end

  defp body(name, args, _kwargs, extended_body, single_spec, kwarg_spec, doc) when length(args) > 0 do
    quote do
      # @spec unquote(name)(any(), [{:key, integer()} | {:key2, integer()}]) :: ExTorch.Tensor.t()
      @doc unquote(doc)
      unquote(kwarg_spec)
      def unquote(name)(unquote_splicing(args), unquote(Macro.var(:kwargs, nil))) do
        unquote(extended_body)
      end

      # @doc false
      # @spec unquote(name)(any()) :: ExTorch.Tensor.t()
      unquote(single_spec)
      def unquote(name)(unquote_splicing(args)) do
        unquote(Macro.var(:kwargs, nil)) = %{}
        unquote(extended_body)
      end
    end
  end

  defp body(name, args, _, extended_body, _, _, _) when length(args) == 0 do
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

  defmacro deftensor(fn_name_args, do: fn_body) do
    docs = get_doc_map()
    typespecs = get_specs()

    {name, _, args} = fn_name_args
    :logger.debug("#{inspect(args)}")
    kwargs = Enum.filter(args, fn e -> match?({:\\, _, _}, e) end)
    args = Enum.filter(args, fn e -> !match?({:\\, _, _}, e) end)

    kwargs = Enum.map(kwargs, &kwarg_map/1)
    # fn_kwargs = Enum.reduce(kwargs, %{}, fn x, acc -> Map.merge(x, acc) end)
    fn_kwargs = Enum.into(kwargs, %{})

    :logger.debug("#{inspect(args)}")
    :logger.debug("#{inspect(kwargs)}")
    :logger.debug("#{inspect(fn_kwargs)}")

    :logger.debug("#{inspect(Map.get(typespecs, name, nil))}")
    :logger.debug("#{inspect(Map.get(docs, name, ""))}")

    kwarg_names = Enum.map(kwargs, fn {kwarg_name, _} -> kwarg_name end)
    func_typespec = Map.get(typespecs, name, nil)
    {single_spec, kwarg_spec} = get_new_specs(args, kwarg_names, func_typespec)

    func_doc = Map.get(docs, name, "")

    :logger.debug("#{inspect single_spec}")
    :logger.debug("#{inspect kwarg_spec}")
    :logger.debug("#{Macro.to_string(single_spec)}")
    :logger.debug("#{Macro.to_string(kwarg_spec)}")


    # {:"::", [line: 145],
    #  [
    #    {:zeros, [line: 145],
    #     [
    #       {:|, [line: 138], [{:tuple, [line: 138], []}, [{:integer, [line: 138], []}]]},
    #       {{:., [line: 139], [ExTorch.DType, :dtype]}, [line: 139], []},
    #       {{:., [line: 140], [ExTorch.Layout, :layout]}, [line: 140], []},
    #       {{:., [line: 141], [ExTorch.Device, :device]}, [line: 141], []},
    #       {:boolean, [line: 142], []},
    #       {:boolean, [line: 143], []},
    #       {{:., [line: 144], [ExTorch.MemoryFormat, :memory_format]}, [line: 144], []}
    #     ]},
    #    {{:., [line: 145], [ExTorch.Tensor, :t]}, [line: 145], []}
    #  ]}

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

    actual_body = body(name, args, kwarg_names, new_body, single_spec, kwarg_spec, func_doc)
    :logger.debug("#{Macro.to_string(Macro.expand(actual_body, nil))}")
    actual_body
  end

  defmacro native_calls(do: body) do
    {:__block__, _, native_defs} = body

    native_module = {:__aliases__, [alias: false], [:ExTorch, :Native]}

    redefinitions =
      Enum.map(native_defs, fn fun_def = {fun_name, _, fun_args} ->
        all_args = get_args(fun_args)
        native_call = {:., [], [native_module, fun_name]}

        quote do
          deftensor unquote(fun_def) do
            unquote(native_call)(unquote_splicing(all_args))
          end
        end
      end)

    {:__block__, [], redefinitions}
  end
end
