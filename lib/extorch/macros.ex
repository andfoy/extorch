defmodule ExTorch.Macros do
  defp kwarg_map({:\\, _, [{arg, _, _, }, value]}) do
    %{arg => value}
  end

  defp body(name, args, extended_body) when length(args) > 0 do
    quote do
      def unquote(name) (unquote_splicing(args), unquote(Macro.var(:kwargs, nil))) do
        unquote(extended_body)
      end
      def unquote(name) (unquote_splicing(args)) do
        unquote(Macro.var(:kwargs, nil)) = %{}
        unquote(extended_body)
      end
    end
  end

  defp body(name, args, extended_body) when length(args) == 0 do
    quote do
      def unquote(name) (unquote(Macro.var(:kwargs, nil))) do
        unquote(extended_body)
      end
      def unquote(name) () do
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
    {name, _, args} = fn_name_args

    kwargs = Enum.filter(args, fn(e) -> match?({:\\, _, _}, e) end)
    args = Enum.filter(args, fn(e) -> !match?({:\\, _, _}, e) end)
    kwargs = Enum.map(kwargs, &kwarg_map/1)
    fn_kwargs = Enum.reduce(kwargs, %{}, fn x, acc -> Map.merge(x, acc) end)

    fn_kwargs_var = Macro.escape(fn_kwargs)
    args_reassignment = Enum.map(fn_kwargs, fn {k, _} -> {k, Macro.var(k, nil)} end)
    kwargs_assignment = {:%{}, [], args_reassignment}

    new_body = quote do
      unquote(Macro.var(:kwargs, nil)) = Enum.into(
        unquote(Macro.var(:kwargs, nil)), %{})
      unquote(Macro.var(:kwargs, nil)) = Map.merge(
        unquote(fn_kwargs_var), unquote(Macro.var(:kwargs, nil)))
      unquote(kwargs_assignment) = unquote(Macro.var(:kwargs, nil))

      unquote(Macro.var(:tensor_result, nil)) = unquote(fn_body)
      ExTorch.Tensor.wrap_tensor_ref(unquote(Macro.var(:tensor_result, nil)))
    end
    actual_body = body(name, args, new_body)
    :logger.debug("#{Macro.to_string(Macro.expand(actual_body, nil))}")
    actual_body
  end
end
