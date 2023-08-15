defmodule ExTorch.MixProject do
  use Mix.Project

  def project do
    [
      app: :extorch,
      version: "0.1.0-pre0",
      elixir: "~> 1.10",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      test_coverage: [
        ignore_modules: [
          ExTorch.DType,
          ExTorch.DelegateWithDocs,
          ExTorch.DelegateWithDocs.Error,
          ExTorch.Device,
          ExTorch.Layout,
          ExTorch.MemoryFormat,
          ExTorch.ModuleMixin,
          ExTorch.Native.BindingDeclaration,
          ExTorch.Native.Macros,
          ExTorch.Index.Slice,
          ExTorch.Utils.ListWrapper,
          Inspect.ExTorch.Tensor,
          Mix.Tasks.PullLibTorch
        ]
      ],
      description: description(),
      package: package()
      # compilers: [:rustler] ++ Mix.compilers(),
      # rustler_crates: [extorch_native: []]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :ssl, :inets],
      env: [
        libtorch: libtorch_config()
      ]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.29.0"},
      {:ex_doc, "~> 0.23", only: :dev, runtime: false},
      {:credo, "~> 1.5", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.3", only: [:dev], runtime: false}
      # {:delegate_with_docs, "~> 0.1.0"}
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end

  defp description do
    "Elixir/Erlang bindings for libtorch."
  end

  defp package do
    [
      # This option is only needed when you don't want to use the OTP application name
      name: "extorch",
      # These are the default files included in the package
      files: ~w(lib priv native .formatter.exs mix.exs README* LICENSE*),
      exclude_patterns: [
        "native/extorch/target",
        "native/extorch/.cargo",
        "priv/native/libtorch",
        "priv/native/libextorch.so",
        "native/extorch/src/native/native.rs.sum"
      ],
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/andfoy/extorch"}
    ]
  end

  defp libtorch_config() do
    [
      version: "2.0.1",
      cuda_versions: [{11, 7}, {11, 8}],
      variant: :auto,
      nightly: false,
      folder: nil
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
        {:"Tensor information", &(&1[:kind] == :tensor_info)},
        {:"Tensor creation", &(&1[:kind] == :tensor_creation)},
        {:"Tensor manipulation", &(&1[:kind] == :tensor_manipulation)},
        {:"Tensor indexing", &(&1[:kind] == :tensor_indexing)}
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
