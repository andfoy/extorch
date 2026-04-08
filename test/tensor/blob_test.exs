defmodule ExTorchTest.Tensor.BlobTest do
  use ExUnit.Case, async: true

  alias ExTorch.Tensor.Blob

  describe "to_blob/1" do
    test "extracts pointer and metadata from a tensor" do
      t = ExTorch.tensor([[1.0, 2.0], [3.0, 4.0]])
      blob = Blob.to_blob(t)

      assert %Blob{} = blob
      assert is_integer(blob.ptr)
      assert blob.ptr > 0
      assert blob.shape == {2, 2}
      assert blob.strides == [2, 1]
      assert blob.dtype == :float
      assert blob.device == :cpu
      assert blob.element_size == 4
      assert blob.owner == t
    end

    test "extracts correct strides for non-square tensors" do
      t = ExTorch.ones({3, 4, 5})
      blob = Blob.to_blob(t)

      assert blob.shape == {3, 4, 5}
      assert blob.strides == [20, 5, 1]
    end

    test "works with different dtypes" do
      t = ExTorch.zeros({2, 3}, dtype: :float64)
      blob = Blob.to_blob(t)

      assert blob.dtype == :double
      assert blob.element_size == 8
    end
  end

  describe "from_blob/2" do
    test "creates a tensor sharing the same memory" do
      original = ExTorch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      blob = Blob.to_blob(original)

      view = Blob.from_blob(blob, owner: original)

      assert %ExTorch.Tensor.BlobView{} = view
      assert view.owner == original
      assert view.tensor.size == {2, 3}
      assert ExTorch.equal(original, view.tensor)

      # Verify same memory address
      assert ExTorch.Native.data_ptr(view.tensor) == blob.ptr
    end

    test "returns bare tensor without owner" do
      original = ExTorch.tensor([1.0, 2.0, 3.0])
      blob = Blob.to_blob(original)

      tensor = Blob.from_blob(blob)

      assert %ExTorch.Tensor{} = tensor
      assert ExTorch.equal(original, tensor)
    end

    test "works with map input" do
      original = ExTorch.randn({4, 4})
      ptr = ExTorch.Native.data_ptr(original)
      strides = ExTorch.Native.strides(original)

      view =
        Blob.from_blob(
          %{ptr: ptr, shape: {4, 4}, dtype: :float, strides: strides},
          owner: original
        )

      assert ExTorch.equal(original, view.tensor)
    end

    test "preserves data across integer dtypes" do
      original = ExTorch.tensor([1, 2, 3, 4, 5], dtype: :int64)
      blob = Blob.to_blob(original)

      view = Blob.from_blob(blob, owner: original)

      assert ExTorch.equal(original, view.tensor)
      assert view.tensor.dtype == :long
    end

    test "owner reference prevents source from being GC eligible" do
      original = ExTorch.tensor([1.0, 2.0, 3.0])
      blob = Blob.to_blob(original)
      view = Blob.from_blob(blob, owner: original)

      # The view holds a reference to original, so even if we lose
      # our local binding, the owner stays alive through the view
      assert view.owner == original
      assert is_struct(view.owner, ExTorch.Tensor)
    end
  end

  describe "data_ptr/1 and strides/1" do
    test "data_ptr returns consistent pointer" do
      t = ExTorch.ones({3, 3})
      ptr1 = ExTorch.Native.data_ptr(t)
      ptr2 = ExTorch.Native.data_ptr(t)
      assert ptr1 == ptr2
    end

    test "strides are correct for contiguous tensor" do
      t = ExTorch.ones({2, 3, 4})
      strides = ExTorch.Native.strides(t)
      assert strides == [12, 4, 1]
    end

    test "element_size matches dtype" do
      assert ExTorch.Native.element_size(ExTorch.ones({1}, dtype: :float32)) == 4
      assert ExTorch.Native.element_size(ExTorch.ones({1}, dtype: :float64)) == 8
      assert ExTorch.Native.element_size(ExTorch.ones({1}, dtype: :int16)) == 2
    end

    test "is_contiguous returns true for standard tensors" do
      t = ExTorch.ones({3, 3})
      assert ExTorch.Native.is_contiguous(t)
    end
  end
end
