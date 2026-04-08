cuda_available =
  try do
    ExTorch.empty({1}, device: :cuda)
    true
  rescue
    _ -> false
  end

exclude = if cuda_available, do: [], else: [:cuda]
ExUnit.start(exclude: exclude)
