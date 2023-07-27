defmodule Mix.Tasks.PullLibTorch do
  @moduledoc "Mix task that downloads libtorch into the current extorch priv directory: `mix help pull_lib_torch`"
  use Mix.Task

  @cuda_regex ~r".*release ((\d+).(\d+)).*"

  defp parse_version_parts([], acc_version, _) do
    List.to_tuple(Enum.reverse(acc_version))
  end

  defp parse_version_parts([part | rest], acc_version, def_version) do
    case Integer.parse(part) do
      {parsed_part, _} ->
        parse_version_parts(rest, [parsed_part | acc_version], def_version)

      :error ->
        def_version
    end
  end

  defp parse_version(version_str, version) do
    case Regex.run(@cuda_regex, version_str) do
      [_, _ | version_parts] ->
        parse_version_parts(version_parts, [], version)

      nil ->
        version
    end
  end

  defp get_cuda_version(port, ref, version) do
    receive do
      {^port, {:data, out}} ->
        version = parse_version(out, version)
        get_cuda_version(port, ref, version)

      {:DOWN, ^ref, :port, ^port, _} ->
        version
    end
  end

  defp download_libtorch(version, cuda_versions, out_folder, nightly) do
    :inets.start()
    :ssl.start()

    # Detect if CUDA is installed
    nvcc = System.find_executable("nvcc")

    torch_flavor =
      case nvcc do
        nil ->
          {:cpu, nil}

        _ ->
          p = Port.open({:spawn_executable, nvcc}, [:binary, args: ["--version"]])
          r = Port.monitor(p)
          get_cuda_version(p, r, {:cpu, nil})
      end

    dist_str =
      case torch_flavor do
        {:cpu, nil} ->
          IO.puts("No CUDA installation was detected, resorting to CPU")
          "cpu"

        {:cuda, cu_version} ->
          {major, minor} = cu_version
          IO.puts("CUDA version detected: #{major}.#{minor}")
          {major, minor} = Enum.max(Enum.filter(cuda_versions, fn v -> v <= cu_version end))
          IO.puts("Closest supported libtorch CUDA version: #{major}.#{minor}")
          "cu#{major}#{minor}"
      end

    url =
      case nightly do
        true ->
          ~c"https://download.pytorch.org/libtorch/nightly/#{dist_str}/libtorch-shared-with-deps-#{version}.zip"

        false ->
          ~c"https://download.pytorch.org/libtorch/#{dist_str}/libtorch-shared-with-deps-#{version}%2B#{dist_str}.zip"
      end

    IO.puts("Downloading #{url} to #{out_folder}")
    libtorch_path = Path.join(out_folder, "libtorch.zip")

    {:ok, :saved_to_file} =
      :httpc.request(:get, {url, []}, [], stream: String.to_charlist(libtorch_path))

    IO.puts("Success!")
    IO.puts("Extracting library")

    {:ok, _} =
      :zip.extract(
        String.to_charlist(libtorch_path),
        [{:cwd, String.to_charlist(out_folder)}, :verbose]
      )

    File.rm(libtorch_path)
  end

  defp get_version_and_nightly(parsed, version, nightly) do
    case nightly do
      true ->
        {"latest", nightly}

      false ->
        ver = Keyword.get(parsed, :version, version)

        case ver do
          "latest" -> {"latest", true}
          "nightly" -> {"latest", true}
          _ -> {ver, false}
        end
    end
  end

  defp get_cuda_versions(parsed, cuda_versions) do
    cuda_versions = Keyword.get(parsed, :cuda, cuda_versions)

    case cuda_versions do
      [_ | _] ->
        cuda_versions

      _ ->
        tuple_ver =
          cuda_versions
          |> String.split(".")
          |> Enum.map(fn x ->
            {num, _} = Integer.parse(x)
            num
          end)
          |> List.to_tuple()

        [tuple_ver]
    end
  end

  @shortdoc "Download libtorch into the current extorch priv directory"
  def run(args) do
    {parsed, _, _} =
      OptionParser.parse(args, strict: [version: :string, cuda: :string, nightly: :boolean])

    folder = to_string(:code.priv_dir(:extorch))

    folder =
      folder
      |> Path.join("native")

    config = Mix.Project.config()
    version = Keyword.get(config, :libtorch_version)
    cuda_versions = Keyword.get(config, :libtorch_cuda_versions)
    libtorch_loc = Path.join(folder, "libtorch")

    nightly = Keyword.get(parsed, :nightly, false)

    {version, nightly} = get_version_and_nightly(parsed, version, nightly)
    cuda_versions = get_cuda_versions(parsed, cuda_versions)

    case File.exists?(libtorch_loc) do
      true ->
        IO.puts("libtorch already exists!")
        :ok

      false ->
        IO.puts("Attempting to download libtorch v#{version}")
        download_libtorch(version, cuda_versions, folder, nightly)
    end
  end
end
