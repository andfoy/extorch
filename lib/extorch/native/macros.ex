defmodule ExTorch.Native.Macros do
  @moduledoc """
  General purpose macros to automatically generate binding declarations and calls
  for both ExTorch callable functions and Rustler signature calls to the NIF library.
  """

  @doc """
  Automatic binding generation.

  This macro allows to define a bindings block under a given `doc_section`
  for a given set of function bindings. All binding declarations should be
  signaled using the `defbinding` function, which recieves the function
  signature, alongside an optional keyword list of parameter transformations
  that must be done before calling the native function
  (defined in `ExTorch.Native`).

  Each `defbinding` declaration must declare its `@spec` and optionally its
  docstring `@doc` before the call. Additionally, the function binding
  signature can declare optional arguments. For example:

        # All function docstrings will be collected under :doc_section_name
        defbindings(:doc_section_name) do
          @doc \"\"\"
          The docstring for func goes here
          \"\"\"
          @spec func(type_1(), type_2(), type_3()) :: ret_type()
          defbinding(
            func(
              arg1,                    # Positional argument
              arg2 \\\\ optional_value,  # Optional argument
              arg3 \\\\ optional_value   # Optional argument
            )
          )

          @doc \"\"\"
          The docstring for func2 goes here
          \"\"\"
          @spec func2(type_1(), type_2(), type_3(), type_4()) :: ret_type()
          defbinding(
            func2(
              arg1 \\\\ optional_value,  # Positional argument with optional value
              arg2,                    # Positional argument
              arg3 \\\\ optional_value,  # Optional argument
              arg4 \\\\ optional_value   # Optional argument
            )
          )

          @doc \"\"\"
          The docstring for func3 goes here
          \"\"\"
          @spec func3(type_1(), type_2(), type_3(), type_4()) :: ret_type()
          defbinding(
            func3(
              arg1,                    # Positional argument
              arg2,                    # Positional argument
              arg3 \\\\ optional_value,  # Optional argument
              arg4 \\\\ optional_value   # Optional argument
            ),
            arg1: arg1[:value],
            arg3: call_to_some_transform(arg3, arg2),
          )
        end

  In case optional arguments are defined, the macro will expand the declaration
  to allow optional arguments to be passed as a keyword list. For example, the
  function `func` will be expanded to the following function calls: `func(arg1)`,
  `func(arg1, kwargs)`, `func(arg1, arg2)`, `func(arg1, arg2, kwargs)` and
  `func(arg1, arg2, arg3)`, where kwargs correspond to `arg2: value,
  arg3: value2` and `arg3: value`, respectively.

  When the first argument is declared as optional, the macro will
  generate function calls that begin with the first argument as well as the
  second argument. In case there are multiple calls with the same arity, the
  macro will try to disambiguate them by computing the corresponding guards
  that distinguish each call from the others. In the case of `func2`, the
  expanded definitions would correspond to `func2(arg2)` `func2(arg2, kwargs)`,
  `func2(arg2, arg3, kwargs)`, `func2(arg2, arg3, arg4)`,
  `func2(arg1, arg2)`, `func2(arg1, arg2, kwargs)`,
  `func2(arg1, arg2, arg3)`, etc.

  Finally, if transforms are defined (like `func3`), they will be assigned to
  the specified arguments before calling the native function.
  """
  defmacro defbindings(doc_section, [{:do, {:__block__, [], args}}]) do
    block = compose_block(args, [], [], doc_section)
    {:__block__, [], block}
  end

  defp compose_block([], block, attrs, _) do
    attrs = Enum.reverse(attrs)
    block = Enum.reverse(block)
    block ++ attrs
  end

  defp compose_block([{:@, _, _} = attr | rest], block, attrs, doc_section) do
    compose_block(rest, block, [attr | attrs], doc_section)
  end

  defp compose_block([{:defbinding, _, [call]} | rest], block, attrs, doc_section) do
    expanded_definition = expand_binding(call, attrs, doc_section, [])
    compose_block(rest, [expanded_definition | block], [], doc_section)
  end

  defp compose_block([{:defbinding, _, [call | transforms]} | rest], block, attrs, doc_section) do
    expanded_definition = expand_binding(call, attrs, doc_section, Enum.at(transforms, 0))
    compose_block(rest, [expanded_definition | block], [], doc_section)
  end

  defp compose_block([head | rest], block, attrs, doc_section) do
    block = attrs ++ block
    compose_block(rest, [head | block], [], doc_section)
  end

  defp compose_potential_middle_signatures(func_name, arg_types, arg_info, transforms) do
    {args, kwargs, derived_kwargs, defaults, arg_pos} = split_args_kwargs(arg_info)

    disjoint_kwargs =
      Enum.filter(kwargs, fn kw ->
        %{^kw => pos} = arg_pos
        pos < length(args)
      end)

    case disjoint_kwargs do
      [] ->
        {args, kwargs, derived_kwargs, defaults, []}

      [kw] ->
        %{^kw => kw_pos} = arg_pos
        %{^kw => default_value} = defaults
        extended_args = List.insert_at(args, kw_pos, default_value)
        [_ | partial_kwargs] = kwargs

        signatures =
          compute_signatures(
            func_name,
            arg_types,
            extended_args,
            partial_kwargs,
            defaults,
            transforms,
            derived_kwargs,
            []
          )

        new_args = List.insert_at(args, kw_pos, kw)
        {new_args, partial_kwargs, derived_kwargs, defaults, signatures}

      _ ->
        raise "there can only be a single optional argument in the middle of all arguments"
    end
  end

  defp expand_binding({func_name, _, args}, attrs, doc_section, transforms) do
    func_info = collect_function_info(attrs, %{:doc => nil, :spec => nil})
    %{:spec => spec, :doc => func_docstring} = func_info

    case spec do
      nil -> raise "@spec declaration is missing for #{func_name}"
      _ -> nil
    end

    {arg_names, arg_info} = collect_arg_info(args)
    {ret_type, arg_types} = collect_arg_types(func_name, spec, arg_names)
    {transforms, aliases, debug} = assemble_transforms(transforms)

    {args, kwargs, derived_kwargs, defaults, first_optional_signatures} =
      compose_potential_middle_signatures(
        func_name,
        arg_types,
        arg_info,
        transforms
      )

    full_positional_signatures =
      compute_signatures(
        func_name,
        arg_types,
        args,
        kwargs,
        defaults,
        transforms,
        derived_kwargs,
        []
      )

    all_signatures = first_optional_signatures ++ full_positional_signatures

    signature_map = Enum.map(all_signatures, fn %{:signature => sig} = x -> {sig, x} end)
    signature_map = Enum.into(signature_map, %{})

    arity_map =
      Enum.reduce(signature_map, %{}, fn {k, %{:arity => arity}}, acc ->
        arity_funcs = Map.get(acc, arity, [])
        Map.put(acc, arity, [k | arity_funcs])
      end)

    max_arity = Enum.max(Map.keys(arity_map))

    valid_signatures =
      Enum.reduce(arity_map, [], fn {_, signatures}, to_generate ->
        valid_arity_signatures = compare_and_reduce_signatures(signatures, arg_types)
        to_generate ++ valid_arity_signatures
      end)

    arity_docs =
      Enum.reduce(valid_signatures, %{}, fn {sig, _}, acc ->
        arity = length(sig)
        arity_funcs = Map.get(acc, arity, [])
        sig_str = Enum.map_join(sig, ", ", fn arg -> Atom.to_string(arg) end)
        sig_str = "* `#{func_name}(#{sig_str})`"
        Map.put(acc, arity, [sig_str | arity_funcs])
      end)

    arity_docs =
      Enum.map(arity_docs, fn {k, v} ->
        doc = """
        Available signature calls:

        #{Enum.join(v, "\n")}
        """

        {k, doc}
      end)

    arity_docs = Enum.into(arity_docs, %{})

    signature_bodies =
      compose_binding_call(
        {func_name, nil, debug},
        doc_section,
        func_docstring,
        ret_type,
        max_arity,
        arity_docs,
        valid_signatures,
        signature_map
      )

    Enum.reduce(aliases, signature_bodies, fn alias_, sigs ->
      alias_bodies =
        compose_binding_call(
          {alias_, func_name, debug},
          doc_section,
          func_docstring,
          ret_type,
          max_arity,
          arity_docs,
          valid_signatures,
          signature_map
        )

      sigs ++ alias_bodies
    end)
  end

  defp get_alias_docstring(original_alias, arity, func_docstring) do
    case original_alias do
      nil ->
        case is_binary(func_docstring) do
          true ->
            quote do
              @doc unquote(func_docstring)
            end

          false ->
            func_docstring
        end

      _ ->
        str = "Alias to `#{original_alias}/#{arity}`"

        quote do
          @doc unquote(str)
        end
    end
  end

  defp compose_binding_call(
         {func_name, original_alias, debug},
         doc_section,
         func_docstring,
         ret_type,
         max_arity,
         arity_docs,
         signatures,
         signature_map
       ) do
    Enum.map(signatures, fn {signature, guards} ->
      guards =
        guards
        |> MapSet.new()
        |> Enum.into([])
        |> compose_guards()

      %{^signature => %{:arity => arity, :body => sig_body, :spec => sig_spec}} = signature_map

      spec =
        quote do
          @spec unquote(func_name)(unquote_splicing(sig_spec)) :: unquote(ret_type)
        end

      func_docstring = get_alias_docstring(original_alias, max_arity, func_docstring)

      doc_headers =
        case {func_docstring, guards} do
          {_docstring, []} when arity == max_arity ->
            quote do
              @doc kind: unquote(doc_section)
              unquote(func_docstring)
            end

          _ ->
            arity_doc = get_alias_docstring(original_alias, arity, Map.get(arity_docs, arity))

            quote do
              @doc kind: unquote(doc_section)
              unquote(arity_doc)
            end
        end

      signature = Enum.map(signature, fn arg -> Macro.var(arg, nil) end)

      body =
        case guards do
          [] ->
            quote do
              unquote(doc_headers)
              unquote(spec)

              def unquote(func_name)(unquote_splicing(signature)) do
                unquote(sig_body)
              end
            end

          _ ->
            quote do
              unquote(doc_headers)
              unquote(spec)

              def unquote(func_name)(unquote_splicing(signature)) when unquote(guards) do
                unquote(sig_body)
              end
            end
        end

      case debug do
        true -> :logger.debug("#{Macro.to_string(body)}")
        false -> nil
      end

      body
    end)
  end

  defp compose_guards(guards) do
    chunk_fun = fn element, acc ->
      if length(acc) == 1 do
        {:cont, {:and, [{:context, Elixir}, {:import, Kernel}], Enum.reverse([element | acc])},
         []}
      else
        {:cont, [element | acc]}
      end
    end

    after_fun = fn
      [] -> {:cont, []}
      acc -> {:cont, Enum.reverse(acc), []}
    end

    composed_guards =
      guards
      |> Enum.map(fn
        {variable, guard} when is_atom(guard) ->
          guard_call = String.to_atom("is_#{Atom.to_string(guard)}")

          quote do
            unquote(guard_call)(unquote(Macro.var(variable, nil)))
          end

        {variable, [_ | _] = guards} ->
          guards
          |> Keyword.keys()
          |> Enum.map(fn guard ->
            guard_call = String.to_atom("is_#{Atom.to_string(guard)}")

            quote do
              unquote(guard_call)(unquote(Macro.var(variable, nil)))
            end
          end)
          |> Enum.reduce(fn guard, acc ->
            quote do
              unquote(guard) or unquote(acc)
            end
          end)

        {variable, {:., _, [struct_alias, :t]}} ->
          quote do
            is_struct(unquote(Macro.var(variable, nil)), unquote(struct_alias))
          end

        {variable, {:atom_list, atom_list}} ->
          quote do
            unquote(Macro.var(variable, nil)) in unquote(atom_list)
          end
      end)
      |> Enum.chunk_while([], chunk_fun, after_fun)

    case composed_guards do
      [h | _] when is_list(h) ->
        [guard | _] = h
        guard

      [h | _] when is_tuple(h) ->
        h

      [] ->
        composed_guards
    end
  end

  defp compare_and_reduce_signatures(signatures, arg_types) do
    Enum.reduce(signatures, [], fn sig, valid_signatures ->
      {valid, guards} = compute_guards_for_signature(valid_signatures, sig, arg_types)

      case valid do
        true -> [{sig, guards} | valid_signatures]
        false -> valid_signatures
      end
    end)
  end

  defp compute_guards_for_signature(valid_signatures, sig, arg_types) do
    Enum.reduce(valid_signatures, {true, []}, fn {valid_sig, _}, {is_valid, guards} ->
      case is_valid do
        true ->
          {valid, diff_sig_guard} = compare_signature_types(sig, valid_sig, arg_types)
          is_valid = valid and is_valid
          guards = guards ++ diff_sig_guard
          {is_valid, guards}

        false ->
          {is_valid, guards}
      end
    end)
  end

  defp compare_signature_types(signature, to_compare_sig, arg_types) do
    sig_arg_types = gather_signature_types(signature, arg_types)
    compare_arg_types = gather_signature_types(to_compare_sig, arg_types)

    sig_arg_types
    |> Enum.zip(Enum.reverse(signature))
    |> Enum.zip(compare_arg_types)
    |> Enum.reduce({false, []}, fn
      {{that_arg, _}, that_arg}, {false, guards} ->
        {false, guards}

      {{this_arg, arg_name}, _that_arg}, {false, guards} ->
        {true, [{arg_name, this_arg} | guards]}

      {_, _}, {true, guards} ->
        {true, guards}
    end)
  end

  defp gather_signature_types(signature, arg_types) do
    signature
    |> Enum.map(fn
      :kwargs ->
        :list

      arg ->
        {type_alias, _} = Map.get(arg_types, arg)
        type_alias
    end)
    |> Enum.reverse()
  end

  defp asm_defaults_macro(defaults, derived_kwargs) do
    defaults
    |> Enum.flat_map(fn {k, v} ->
      case Map.has_key?(derived_kwargs, k) do
        true ->
          %{^k => struct_args} = defaults
          Enum.into(struct_args, [])

        false ->
          [{k, v}]
      end
    end)
  end

  defp asm_call_macro(func_name, args, kwargs, output_transform) do
    native_module = {:__aliases__, [alias: false], [:ExTorch, :Native]}
    call_unquote = {:., [], [native_module, func_name]}

    call_parameters =
      Enum.map(
        args ++ kwargs,
        fn
          x when is_atom(x) -> Macro.var(x, nil)
          x -> x
        end
      )

    call =
      quote do
        unquote(call_unquote)(unquote_splicing(call_parameters))
      end

    case output_transform do
      nil ->
        call

      _ ->
        quote do
          unquote(output_transform)(unquote(call))
        end
    end
  end

  defp parse_struct_spec(kw, kwarg_type) do
    case kwarg_type do
      {{:., _, [{:__aliases__, _, struct_name}, :t]}, _, []} ->
        struct_name = [:"Elixir"] ++ struct_name

        struct_module =
          struct_name
          |> Enum.join(".")
          |> String.to_atom()

        {:ok, [{:type, types}]} = Code.Typespec.fetch_types(struct_module)

        {:"::", [],
         [
           {:t, [], []},
           {:%, _,
            [
              ^struct_module,
              {:%{}, _, struct_spec}
            ]}
         ]} = Code.Typespec.type_to_quoted(types)

        struct_spec

      _ ->
        [{kw, kwarg_type}]
    end
  end

  defp asm_kwargs_spec_macro(kwargs, arg_types, derived_kwargs) do
    Enum.reduce(kwargs, [], fn kw, acc ->
      case Map.has_key?(derived_kwargs, kw) do
        true ->
          {_, kwarg_type} = Map.get(arg_types, kw)
          acc ++ parse_struct_spec(kw, kwarg_type)

        false ->
          {_, kwarg_type} = Map.get(arg_types, kw)
          acc ++ [{kw, kwarg_type}]
      end
    end)
  end

  defp compute_signatures(
         func_name,
         arg_types,
         args,
         [],
         _,
         {transforms, output_transform, omitted_args},
         _,
         signatures
       ) do
    valid_args = Enum.filter(args, fn x -> is_atom(x) and Map.has_key?(arg_types, x) end)
    arity = length(valid_args)
    native_module = {:__aliases__, [alias: false], [:ExTorch, :Native]}
    call_unquote = {:., [], [native_module, func_name]}

    call_args = Enum.filter(args, fn x -> not MapSet.member?(omitted_args, x) end)

    call_parameters =
      Enum.map(
        call_args,
        fn
          x when is_atom(x) -> Macro.var(x, nil)
          x -> x
        end
      )

    fn_spec =
      args
      |> Enum.filter(fn x -> is_atom(x) and Map.has_key?(arg_types, x) end)
      |> Enum.map(fn
        arg ->
          {_, spec_type} = Map.get(arg_types, arg)
          spec_type
      end)

    call =
      quote do
        unquote(call_unquote)(unquote_splicing(call_parameters))
      end

    body =
      case output_transform do
        nil ->
          quote do
            unquote(transforms)
            unquote(call)
          end

        _ ->
          quote do
            unquote(transforms)
            unquote(output_transform)(unquote(call))
          end
      end

    sig_info = %{:signature => valid_args, :arity => arity, :body => body, :spec => fn_spec}
    [sig_info | signatures]
  end

  defp compute_signatures(
         func_name,
         arg_types,
         args,
         [kwarg | rest] = kwargs,
         defaults,
         {transforms, output_transform, omitted_args},
         derived_kwargs,
         signatures
       ) do
    call_args = Enum.filter(args, fn x -> not MapSet.member?(omitted_args, x) end)
    valid_args = Enum.filter(args, fn x -> is_atom(x) and Map.has_key?(arg_types, x) end)
    defaults_macro = asm_defaults_macro(defaults, derived_kwargs)

    defaults_macro = {:%{}, [], defaults_macro}

    kwargs_assignment =
      kwargs
      |> Enum.flat_map(fn kw ->
        case Map.has_key?(derived_kwargs, kw) do
          true -> []
          false -> [{kw, Macro.var(kw, nil)}]
        end
      end)

    deriv_kwarg_unpack =
      derived_kwargs
      |> Enum.map(fn {kwarg, struct_name} ->
        quote do
          unquote(Macro.var(kwarg, nil)) =
            struct(unquote(struct_name), unquote(Macro.var(:kwargs, nil)))

          unquote(Macro.var(kwarg, nil)) =
            ExTorch.Protocol.DefaultStruct.replace_defaults(unquote(Macro.var(kwarg, nil)))
        end
      end)

    deriv_kwarg_unpack = {:__block__, [], deriv_kwarg_unpack}

    kwarg_unpack =
      case kwargs_assignment do
        [] ->
          nil

        _ ->
          kwargs_assignment = {:%{}, [], kwargs_assignment}

          quote do
            unquote(kwargs_assignment) = unquote(Macro.var(:kwargs, nil))
          end
      end

    args_spec =
      valid_args
      |> Enum.map(fn
        arg ->
          {_, spec_type} = Map.get(arg_types, arg)
          spec_type
      end)

    kwargs_spec = asm_kwargs_spec_macro(kwargs, arg_types, derived_kwargs)
    fn_spec = args_spec ++ [kwargs_spec]

    call_kwargs = Enum.filter(kwargs, fn x -> not MapSet.member?(omitted_args, x) end)
    call = asm_call_macro(func_name, call_args, call_kwargs, output_transform)

    kwarg_body =
      quote do
        unquote(Macro.var(:kwargs, nil)) =
          Enum.into(
            unquote(Macro.var(:kwargs, nil)),
            %{}
          )

        unquote(Macro.var(:kwargs, nil)) =
          Map.merge(unquote(defaults_macro), unquote(Macro.var(:kwargs, nil)))

        unquote(kwarg_unpack)
        unquote(deriv_kwarg_unpack)
        unquote(transforms)
        unquote(call)
      end

    body =
      case kwargs_assignment do
        [] ->
          quote do
            unquote(Macro.var(:kwargs, nil)) = unquote(defaults_macro)
            unquote(deriv_kwarg_unpack)
            unquote(transforms)
            unquote(call)
          end

        _ ->
          quote do
            unquote(Macro.var(:kwargs, nil)) = unquote(defaults_macro)
            unquote(kwarg_unpack)
            unquote(deriv_kwarg_unpack)
            unquote(transforms)
            unquote(call)
          end
      end

    kwarg_signature = valid_args ++ [:kwargs]
    kwarg_arity = length(kwarg_signature)

    kwarg_sig_info = %{
      :signature => kwarg_signature,
      :arity => kwarg_arity,
      :body => kwarg_body,
      :spec => fn_spec
    }

    signature = valid_args
    arity = length(signature)
    sig_info = %{:signature => signature, :arity => arity, :body => body, :spec => args_spec}
    signatures = [sig_info | [kwarg_sig_info | signatures]]
    args = args ++ [kwarg]
    {_, defaults} = Map.pop(defaults, kwarg)

    compute_signatures(
      func_name,
      arg_types,
      args,
      rest,
      defaults,
      {transforms, output_transform, omitted_args},
      derived_kwargs,
      signatures
    )
  end

  defp split_args_kwargs(arg_info) do
    {args, kwargs, derived_kwargs, defaults, arg_pos} =
      arg_info
      |> Enum.reduce(
        {[], [], %{}, %{}, %{}},
        fn
          {arg, true, {:%, _, [{:__aliases__, _, struct_parts}, {:%{}, _, set_values}]}},
          {args, kwargs, derived_kwargs, default_values, arg_pos} ->
            struct_name =
              struct_parts
              |> Enum.reduce(["Elixir"], fn x, acc -> [Atom.to_string(x) | acc] end)
              |> Enum.reverse()
              |> Enum.join(".")
              |> String.to_atom()

            {_, default_struct_values} = Map.pop(struct_name.__struct__, :__struct__)
            set_values = Enum.into(set_values, %{})
            default_struct_values = Map.merge(default_struct_values, set_values)

            {args, [arg | kwargs], Map.put(derived_kwargs, arg, struct_name),
             Map.put(default_values, arg, default_struct_values),
             Map.put(arg_pos, arg, map_size(arg_pos))}

          {arg, true, default}, {args, kwargs, derived_kwargs, default_values, arg_pos} ->
            {args, [arg | kwargs], derived_kwargs, Map.put(default_values, arg, default),
             Map.put(arg_pos, arg, map_size(arg_pos))}

          {arg, false, _}, {args, kwargs, derived_kwargs, default_values, arg_pos} ->
            {[arg | args], kwargs, derived_kwargs, default_values,
             Map.put(arg_pos, arg, map_size(arg_pos))}
        end
      )

    args = Enum.reverse(args)
    kwargs = Enum.reverse(kwargs)
    {args, kwargs, derived_kwargs, defaults, arg_pos}
  end

  defp collect_function_info([], acc) do
    acc
  end

  defp collect_function_info([{:@, _, [{:doc, _, _}]} = attr | attrs], acc) do
    acc = Map.put(acc, :doc, attr)
    collect_function_info(attrs, acc)
  end

  defp collect_function_info([{:@, _, [{:spec, _, specs}]} | attrs], acc) do
    acc = Map.put(acc, :spec, specs)
    collect_function_info(attrs, acc)
  end

  defp collect_arg_info(args) do
    args
    |> Enum.map(fn
      {:\\, _, [{arg_name, _, _}, default_value]} ->
        {arg_name, {arg_name, true, default_value}}

      {arg_name, _, _} ->
        {arg_name, {arg_name, false, nil}}
    end)
    |> Enum.unzip()
  end

  defp collect_arg_types(
         func_name,
         [{:"::", _, [{func_name, _, arg_specs}, ret_type]}],
         arg_names
       ) do
    arg_types =
      arg_names
      |> Enum.zip(arg_specs)
      |> Enum.map(&extract_arg_type/1)
      |> Enum.into(%{})

    {ret_type, arg_types}
  end

  defp flatten_type_union([], acc) do
    Enum.reverse(acc)
  end

  defp flatten_type_union([{:|, _, more_union} | rest], acc) do
    flatten_type_union(more_union ++ rest, acc)
  end

  defp flatten_type_union([any | rest], acc) do
    flatten_type_union(rest, [any | acc])
  end

  defp extract_arg_type({arg_name, arg_spec}) do
    case arg_spec do
      {:|, _, type_union} ->
        type_union = flatten_type_union(type_union, [])
        all_atom = Enum.reduce(type_union, true, fn x, acc -> acc and is_atom(x) end)

        type_union =
          case all_atom do
            true -> {:atom_list, type_union}
            false -> parse_type_union(type_union)
          end

        {arg_name, {type_union, arg_spec}}

      _ ->
        type = parse_type(arg_spec)
        {arg_name, type}
    end
  end

  defp parse_type_union(type_union) do
    Enum.map(type_union, &parse_type/1)
  end

  defp parse_type(type) do
    type_alias =
      case type do
        {{:., _, [{:__aliases__, _, [:ExTorch, :Tensor]}, :t]}} -> :tensor
        {{:., _, [{:__aliases__, _, [:ExTorch, :Scalar]}, :t]}, _, []} -> :scalar
        {{:., _, [{:__aliases__, _, [:ExTorch, _]}, extorch_type]}, _, []} -> extorch_type
        {_, _} -> :tuple
        nil -> nil
        {type, _, _} -> type
        [_] -> :list
      end

    {type_alias, type}
  end

  defp assemble_transforms(transforms) do
    {transforms, aliases, output_transform, omitted_args, debug} =
      Enum.reduce(transforms, {[], [], nil, MapSet.new(), false}, fn
        {:output, {transform, [no_parens: true, line: _], _}},
        {transforms, aliases, _, omit, debug} ->
          {transforms, aliases, transform, omit, debug}

        {:output, transform}, {transforms, aliases, _, omit, debug} ->
          {transforms, aliases, transform, omit, debug}

        {:extra, transform}, {transforms, aliases, output_transform, omit, debug} ->
          {[transform | transforms], aliases, output_transform, omit, debug}

        {:fn_aliases, aliases}, {transforms, _, output_transform, omit, debug} ->
          {transforms, aliases, output_transform, omit, debug}

        {:debug_gen_macro, debug}, {transforms, aliases, output_transform, omit, _} ->
          {transforms, aliases, output_transform, omit, debug}

        {:omitted_args, omit}, {transforms, aliases, output_transform, _, debug} ->
          {transforms, aliases, output_transform, MapSet.new(omit), debug}

        {variable, transform}, {transforms, aliases, output_transform, omit, debug} ->
          transform =
            quote do
              unquote(Macro.var(variable, nil)) = unquote(transform)
            end

          {[transform | transforms], aliases, output_transform, omit, debug}
      end)

    {{{:__block__, [], Enum.reverse(transforms)}, output_transform, omitted_args}, aliases, debug}
  end
end
