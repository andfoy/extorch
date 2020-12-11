defmodule ExTorchTest do
  use ExUnit.Case
  doctest ExTorch

  test "greets the world" do
    assert ExTorch.hello() == :world
  end
end
