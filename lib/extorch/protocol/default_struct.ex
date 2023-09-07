defprotocol ExTorch.Protocol.DefaultStruct do
  @doc """
  Given a struct filled with default values, return a struct with valid values.
  """
  def replace_defaults(struct_defaults)
end

defimpl ExTorch.Protocol.DefaultStruct, for: Any do
  def replace_defaults(in_struct) do
    in_struct
  end
end
