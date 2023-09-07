defmodule ExTorchTest.RegistryTest do
  use ExUnit.Case

  test "set_default_dtype/1" do
    {:ok, pid} = Agent.start(fn -> ExTorch.get_default_dtype() end)
    assert ExTorch.get_default_dtype() == :float32

    # Assert that default dtypes are independent across processes
    proc_value = Agent.get_and_update(pid, fn val -> {val, ExTorch.set_default_dtype(:float64)} end)
    assert proc_value == :float32
    assert ExTorch.get_default_dtype() == :float32

    proc_value = Agent.get(pid, fn val -> val end)
    assert proc_value == :float64

    # Create a tensor with default value in both processes
    a = ExTorch.empty({3, 4, 5})
    assert a.dtype == :float

    Agent.update(pid, fn _ -> ExTorch.ones({3, 3}) end)
    b = Agent.get(pid, fn t -> t end)
    assert b.dtype == :double
  end

  test "set_default_device/1" do
    nvcc = System.find_executable("nvcc")
    case nvcc do
      nil -> nil
      _ ->
        {:ok, pid} = Agent.start(fn -> ExTorch.get_default_device() end)
        assert ExTorch.get_default_device() == :cpu

        # Assert that default devices are independent across processes
        proc_value = Agent.get_and_update(pid, fn val -> {val, ExTorch.set_default_device(:cuda)} end)
        assert proc_value == :cpu
        assert ExTorch.get_default_device() == :cpu

        proc_value = Agent.get(pid, fn val -> val end)
        assert proc_value == :cuda

        # Create a tensor with default value in both processes
        a = ExTorch.empty({3, 4, 5})
        assert a.device == :cpu

        Agent.update(pid, fn _ -> ExTorch.ones({3, 3}) end)
        b = Agent.get(pid, fn t -> t end)
        assert b.device == {:cuda, 0}
    end
  end
end
