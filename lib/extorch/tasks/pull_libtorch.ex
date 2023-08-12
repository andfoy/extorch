defmodule Mix.Tasks.PullLibTorch do
  @moduledoc "Mix task that downloads libtorch into the current extorch priv directory: `mix help pull_lib_torch`"
  use Mix.Task

  @shortdoc "Download libtorch into the current extorch priv directory"
  def run(args) do
    {parsed, _, _} =
      OptionParser.parse(args, strict: [version: :string, cuda: :string, nightly: :boolean])

    ExTorch.Utils.DownloadTorch.download_torch_binaries(parsed)
  end
end
