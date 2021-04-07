defmodule ExTorch.Device do
  @moduledoc """
  A torch.device is an object representing the device on which a torch.Tensor
  is or will be allocated. The torch.device contains a device type ('cpu' or 'cuda')
  and optional device ordinal for the device type. If the device ordinal is not
  present, this object will always represent the current device for the device type,
  even after ``torch.cuda.set_device()`` is called; e.g., a torch.Tensor constructed
  with device 'cuda' is equivalent to 'cuda:X' where X is the result of
  ``torch.cuda.current_device()``.

  * A torch.Tensor’s device can be accessed via the Tensor.device property.

  * A torch.device can be constructed via a string or via a string and device ordinal
  """

  @typedoc """
  This object will always represent the current device for the device type
  """
  @type atomic_device :: :cpu | :cuda | :hip | :fpga | :vulkan | :msnpu | :xla

  @typedoc """
  The torch.device contains a device type ('cpu' or 'cuda') and optional device ordinal for the device type
  """
  @type composed_device :: {atomic_device(), integer()}

  @typedoc """
  A torch.device is an object representing the device on which a torch.Tensor is or will be allocated.
  The torch.device argument in functions can generally be substituted with a string. This allows for fast prototyping of code.
  """
  @type device :: atomic_device() | composed_device() | binary()

  @devices [:cpu, :cuda, :hip, :fpga, :vulkan, :msnpu, :xla]

  defguard is_device(device)
           when device in @devices or
                  (is_tuple(device) and tuple_size(device) == 2 and elem(device, 0) in @devices and
                     is_integer(elem(device, 1))) or is_binary(device)
end
