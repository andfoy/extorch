defmodule ExTorch.Registry.Device do
  @moduledoc false
  require ExTorch.Device

  @after_compile __MODULE__
  def __after_compile__(_env, bytecode) do
    __MODULE__
    |> :code.which()
    |> to_string()
    |> File.write(bytecode)
  end

  @doc """
  Get the default device on which `ExTorch.Tensor` structs are allocated.

  ## Notes
  By default, `ExTorch` will set `:cpu` as the default device.
  """
  @doc kind: :process_values
  @spec get_default_device() :: ExTorch.Device.device()
  def get_default_device() do
    case Registry.values(__MODULE__, :device, self()) do
      [] ->
        Registry.register(__MODULE__, :device, :cpu)
        :cpu
      [device] -> device
    end
  end

  @doc """
  Sets the default device on which `ExTorch.Tensor` structs are allocated.

  This function does not affect factory function calls which are called with an
  explicit `device` argument. Factory calls will be performed as if they were
  passed `device` as an argument.

  The default device is initially `:cpu`. If you set the default tensor device to
  another device (e.g., `:cuda`) without a device index, tensors will be
  allocated on whatever the current device for the device type.
  """
  @doc kind: :process_values
  @spec set_default_device(ExTorch.Device.device()) :: ExTorch.Device.device()
  def set_default_device(device) when ExTorch.Device.is_device(device) do
    case Registry.values(__MODULE__, :device, self()) do
      [] ->
        Registry.register(__MODULE__, :device, device)
      _ ->
        Registry.unregister(__MODULE__, :device)
        Registry.register(__MODULE__, :device, device)
    end

    device
  end
end
