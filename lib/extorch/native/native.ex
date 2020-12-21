defmodule ExTorch.Native do
  use Rustler, otp_app: :extorch, crate: "extorch_native"

  # When your NIF is loaded, it will override this function.
  def add(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  def size(_a), do: :erlang.nif_error(:nif_not_loaded)

  def empty(_sizes, _dtype, _layout, _device, _requires_grad, _pin_memory, _mem_fmt) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def dtype(_a), do: :erlang.nif_error(:nif_not_loaded)

  def device(_a), do: :erlang.nif_error(:nif_not_loaded)
end
