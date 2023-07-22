defmodule ExTorch.MixProject do
  use Mix.Project

  def project do
    [
      app: :extorch,
      version: "0.1.0-dev0",
      elixir: "~> 1.10",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      libtorch_version: "2.0.1",
      libtorch_cuda_versions: [{11, 7}, {11, 8}]
      # compilers: [:rustler] ++ Mix.compilers(),
      # rustler_crates: [extorch_native: []]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :ssl, :inets]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.29.0"},
      {:ex_doc, "~> 0.23", only: :dev, runtime: false},
      {:credo, "~> 1.5", only: [:dev, :test], runtime: false},
      # {:delegate_with_docs, "~> 0.1.0"}
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end

  defp docs do
    [
      main: "ExTorch",
      # You can specify a function for adding
      # custom content to the generated HTML.
      # This is useful for custom JS/CSS files you want to include.
      before_closing_body_tag: &before_closing_body_tag/1,
      groups_for_functions: [
        {:"Tensor information", & &1[:kind] == :tensor_info},
        {:"Tensor creation", & &1[:kind] == :tensor_creation},
        {:"Tensor manipulation", & &1[:kind] == :tensor_manipulation},
      ]
      # ...
    ]
  end

  # In our case we simply add a <script> tag
  # that loads MathJax from CDN and specify the configuration.
  # Once loaded, the script will dynamically turn any LaTeX
  # expressions on the page into SVG images.
  defp before_closing_body_tag(:html) do
    """
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$']],
          displayMath: [['$$','$$']],
        },
      };
    </script>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
