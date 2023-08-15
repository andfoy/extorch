defmodule ExTorch.Utils.DownloadTorch do
  @moduledoc false

  @libtorch_config Application.compile_env(:extorch, :libtorch)
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

  def detect_cuda_installation(config) do
    # Detect if CUDA is installed
    nvcc = System.find_executable("nvcc")

    tag =
      case nvcc do
        nil ->
          IO.puts("No CUDA installation was detected, resorting to CPU")
          {:cpu, nil}

        _ ->
          p = Port.open({:spawn_executable, nvcc}, [:binary, args: ["--version"]])
          r = Port.monitor(p)
          {:cuda, get_cuda_version(p, r, {:cpu, nil})}
      end

    case tag do
      {:cpu, nil} ->
        "cpu"

      {:cuda, {major, minor} = cu_version} ->
        IO.puts("CUDA version detected: #{major}.#{minor}")
        cuda_versions = Keyword.get(config, :cuda_versions)
        {major, minor} = Enum.max(Enum.filter(cuda_versions, fn v -> v <= cu_version end))
        IO.puts("Closest supported libtorch CUDA version: #{major}.#{minor}")
        "cu#{major}#{minor}"
    end
  end

  defp get_variant_tag(config, :auto) do
    case :os.type() do
      {:unix, :darwin} ->
        IO.puts("MacOS only supports CPU")
        {:macos, "cpu"}

      {:unix, :linux} ->
        str_tag = detect_cuda_installation(config)
        {:linux, str_tag}

      {:win32, :nt} ->
        str_tag = detect_cuda_installation(config)
        {:win, str_tag}
    end
  end

  defp get_variant_tag(_, :cpu) do
    case :os.type() do
      {:unix, :darwin} ->
        IO.puts("MacOS only supports CPU")
        {:macos, "cpu"}

      {:unix, :linux} ->
        {:linux, "cpu"}

      {:win32, :nt} ->
        {:win, "cpu"}
    end
  end

  defp download_libtorch(version, {os, dist_str}, out_folder, nightly) do
    :inets.start()
    :ssl.start()

    url_base = "https://download.pytorch.org/libtorch"

    url =
      case nightly do
        true -> "#{url_base}/nightly"
        false -> url_base
      end

    url = "#{url}/#{dist_str}/libtorch"

    url =
      case os do
        :macos -> "#{url}-macos"
        :linux -> "#{url}-shared-with-deps"
        :win -> "#{url}-win-shared-with-deps"
      end

    url = "#{url}-#{version}"

    url =
      case {os, nightly} do
        {:macos, _} -> "#{url}.zip"
        {_, false} -> "#{url}%2B#{dist_str}.zip"
        {_, true} -> "#{url}.zip"
      end

    url = String.to_charlist(url)

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

  defp parse_nightly(version, nightly) do
    case nightly do
      true ->
        {"latest", nightly}

      false ->
        case version do
          "latest" -> {"latest", true}
          "nightly" -> {"latest", true}
          _ -> {version, false}
        end
    end
  end

  defp get_version_and_nightly(version, nightly) do
    case version do
      :local -> {:local, false}
      _ -> parse_nightly(version, nightly)
    end
  end

  defp get_exit_status(port, ref) do
    receive do
      {^port, {:data, _}} ->
        get_exit_status(port, ref)

      {^port, {:exit_status, 0}} ->
        :ok

      {^port, {:exit_status, _}} ->
        {:error, :pytorch_import_failed}

      {:DOWN, ^ref, :port, ^port, _} ->
        {:error, :error_spawning_python}
    end
  end

  defp find_local_pytorch() do
    python = System.find_executable("python")

    case python do
      nil ->
        {:error, :python_not_found}

      _ ->
        p =
          Port.open({:spawn_executable, python}, [
            :binary,
            args: ["-c", "import torch; print(torch.__file__)", :exit_status]
          ])

        r = Port.monitor(p)
        get_exit_status(p, r)
    end
  end

  defp symlink_local_libtorch(config, libtorch_loc) do
    IO.puts("Using local libtorch installation")
    local_folder = Keyword.get(config, :folder)

    case local_folder do
      :python ->
        find_local_pytorch()

      nil ->
        {:error, :missing_libtorch_folder_key}

      _ ->
        File.ln_s(libtorch_loc, local_folder)
    end
  end

  defp try_to_download_libtorch(
         config,
         variant,
         version,
         nightly,
         libtorch_loc,
         folder
       ) do
    case File.exists?(libtorch_loc) do
      true ->
        IO.puts("libtorch already exists!")
        :ok

      false ->
        variant_tag = get_variant_tag(config, variant)
        IO.puts("Attempting to download libtorch v#{version}")
        download_libtorch(version, variant_tag, folder, nightly)
    end
  end

  def download_torch_binaries(config \\ []) do
    folder = to_string(:code.priv_dir(:extorch))

    folder =
      folder
      |> Path.join("native")

    libtorch_loc = Path.join(folder, "libtorch")

    version = Keyword.get(config, :version)
    variant = Keyword.get(config, :variant)
    nightly = Keyword.get(config, :nightly, false)

    {version, nightly} = get_version_and_nightly(version, nightly)

    :ok =
      case version do
        :local ->
          symlink_local_libtorch(config, libtorch_loc)

        _ ->
          :ok
      end

    try_to_download_libtorch(
      config,
      variant,
      version,
      nightly,
      libtorch_loc,
      folder
    )
  end

  defp default_libtorch_config() do
    # 2.0.1 stable release options
    [
      version: "2.0.1",
      cuda_versions: [{11, 7}, {11, 8}],
      variant: :auto,
      nightly: false,
      folder: nil
    ]

    # 2.1.0 Nightly version options
    # [
    #   version: "latest",
    #   cuda_versions: [{11, 8}, {12, 1}],
    #   variant: :auto,
    #   nightly: false,
    #   folder: nil
    # ]
  end
  defmacro __using__(_opts) do
    config = case @libtorch_config do
      nil -> default_libtorch_config()
      _ -> @libtorch_config
    end

    :ok = download_torch_binaries(config)

    quote do
      def compilation_info() do
        unquote(config)
      end
    end
  end
end
