defmodule ExTorch.Application do
  @moduledoc false
  use Application

  def start(_type, _args) do
    children = [
      {Registry, [name: ExTorch.Registry.DType, keys: :duplicate]},
      {Registry, [name: ExTorch.Registry.Device, keys: :duplicate]},
    ]
    Supervisor.start_link(children, strategy: :one_for_one)
  end
end
