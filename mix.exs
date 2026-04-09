defmodule ExTorch.MixProject do
  use Mix.Project

  def project do
    [
      app: :extorch,
      version: "0.2.1-pre0",
      elixir: "~> 1.16",
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
          Inspect.ExTorch.JIT.Model,
          Inspect.ExTorch.NN.Layer,
          Inspect.ExTorch.Tensor.BlobView,
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
      mod: {ExTorch.Application, []},
      extra_applications: [:logger, :ssl, :inets]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.37.3"},
      {:telemetry, "~> 1.2"},
      {:ex_doc, "~> 0.35", only: :dev, runtime: false},
      # {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:phoenix_live_dashboard, "~> 0.8", optional: true}
    ]
  end

  defp description do
    "Production ML model serving on the BEAM. Load TorchScript models, define neural networks with an Elixir DSL, and monitor serving performance with telemetry."
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

  defp docs do
    [
      main: "getting-started",
      extras: [
        "guides/getting-started.md",
        "guides/serving-models.md",
        "guides/neural-network-dsl.md",
        "guides/observability.md"
      ],
      groups_for_extras: [
        Guides: Path.wildcard("guides/*.md")
      ],
      # You can specify a function for adding
      # custom content to the generated HTML.
      # This is useful for custom JS/CSS files you want to include.
      before_closing_body_tag: &before_closing_body_tag/1,
      groups_for_docs: [
        {:"Per-process settings", &(&1[:kind] == :process_values)},
        {:"Tensor information", &(&1[:kind] == :tensor_info)},
        {:"Tensor creation", &(&1[:kind] == :tensor_creation)},
        {:"Tensor manipulation", &(&1[:kind] == :tensor_manipulation)},
        {:"Tensor indexing", &(&1[:kind] == :tensor_indexing)},
        {:"Pointwise math operations", &(&1[:kind] == :tensor_pointwise)},
        {:"Reduction operations", &(&1[:kind] == :tensor_reduction)},
        {:"Comparison operations", &(&1[:kind] == :tensor_comparison)},
        {:"Other operations", &(&1[:kind] == :tensor_other_ops)}
      ],
      groups_for_modules: [
        "General API": [ExTorch, ExTorch.Tensor],
        "JIT Model Serving": [
          ExTorch.JIT,
          ExTorch.JIT.Model,
          ExTorch.JIT.Server
        ],
        "Neural Network": [
          ExTorch.NN,
          ExTorch.NN.Module,
          ExTorch.NN.Layer,
          ExTorch.NN.Introspect,
          ExTorch.NN.Introspect.Schema,
          ExTorch.NN.JITBackedModel
        ],
        "NN Layers": [
          ExTorch.NN.Linear,
          ExTorch.NN.Conv1d,
          ExTorch.NN.Conv2d,
          ExTorch.NN.Conv3d,
          ExTorch.NN.ConvTranspose1d,
          ExTorch.NN.ConvTranspose2d,
          ExTorch.NN.MaxPool1d,
          ExTorch.NN.MaxPool2d,
          ExTorch.NN.AvgPool1d,
          ExTorch.NN.AvgPool2d,
          ExTorch.NN.AdaptiveAvgPool1d,
          ExTorch.NN.AdaptiveAvgPool2d,
          ExTorch.NN.BatchNorm1d,
          ExTorch.NN.BatchNorm2d,
          ExTorch.NN.LayerNorm,
          ExTorch.NN.GroupNorm,
          ExTorch.NN.InstanceNorm1d,
          ExTorch.NN.InstanceNorm2d,
          ExTorch.NN.Dropout,
          ExTorch.NN.Embedding,
          ExTorch.NN.LSTM,
          ExTorch.NN.GRU,
          ExTorch.NN.MultiheadAttention,
          ExTorch.NN.Flatten,
          ExTorch.NN.Unflatten
        ],
        "NN Activations": [
          ExTorch.NN.ReLU,
          ExTorch.NN.LeakyReLU,
          ExTorch.NN.GELU,
          ExTorch.NN.ELU,
          ExTorch.NN.SiLU,
          ExTorch.NN.Mish,
          ExTorch.NN.PReLU,
          ExTorch.NN.Sigmoid,
          ExTorch.NN.Tanh,
          ExTorch.NN.Softmax,
          ExTorch.NN.LogSoftmax
        ],
        "Tensor Exchange": [
          ExTorch.Tensor.Blob,
          ExTorch.Tensor.BlobView
        ],
        "AOTI Compiled Models": [
          ExTorch.AOTI,
          ExTorch.AOTI.Model,
          ExTorch.AOTI.Server
        ],
        "Export Reader": [
          ExTorch.Export,
          ExTorch.Export.Model,
          ExTorch.Export.Server
        ],
        "Observability": [
          ExTorch.Metrics,
          ExTorch.Observer.Dashboard
        ],
        "Exchange types": [
          ExTorch.Complex,
          ExTorch.Index,
          ExTorch.Index.Slice,
          ExTorch.Tensor.Options,
          ExTorch.Utils.PrintOptions,
          ExTorch.Utils.ListWrapper
        ],
        "Spec types": [
          ExTorch.Scalar,
          ExTorch.DType,
          ExTorch.Device,
          ExTorch.Layout,
          ExTorch.MemoryFormat
        ],
        Protocols: [ExTorch.Protocol.DefaultStruct],
        Macros: [
          ExTorch.Native.Macros,
          ExTorch.Native.BindingDeclaration,
          ExTorch.DelegateWithDocs,
          ExTorch.ModuleMixin
        ],
        "Native API": [ExTorch.Native],
        "Other utilities": [ExTorch.Utils, ExTorch.Utils.Types]
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
