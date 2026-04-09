cuda_available =
  try do
    ExTorch.empty({1}, device: :cuda)
    true
  rescue
    _ -> false
  end

exclude = [:popular_models]
exclude = if cuda_available, do: exclude, else: [:cuda | exclude]
ExUnit.start(exclude: exclude)
