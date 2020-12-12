defmodule ExTorch.Native do
  use Rustler, otp_app: :extorch, crate: "extorch_native"

  # When your NIF is loaded, it will override this function.
  def add(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
end
