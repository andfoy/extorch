// #[macro_use]
// use cxx::SharedPtr;
extern crate cxx;

#[cxx::bridge]
pub mod torch {
    /// Shared interface to a tensor pointer in memory.
    struct CrossTensorRef {
        tensor: SharedPtr<CrossTensor>,
    }

    /// Torch tensor device descriptor.
    struct Device {
        device: String,
        index: i64,
    }

    /// Scalar number representation
    struct Scalar {
        _ui8: u8,
        _i8: i8,
        _i16: i16,
        _i32: i32,
        _i64: i64,
        _f16: f32,
        _f32: f32,
        _f64: f64,
        _bool: bool,
        entry_used: String,
    }

    struct ScalarList {
        list: Vec<Scalar>,
        size: Vec<i64>
    }

    extern "Rust" {}

    unsafe extern "C++" {
        include!("extorch_native/include/wrapper.h");

        /// Reference to a torch tensor in memory
        type CrossTensor;

        // Tensor attribute access
        /// Get the size of a tensor
        fn size(tensor: &SharedPtr<CrossTensor>) -> &'static [i64];
        /// Get the type of a tensor
        fn dtype(tensor: &SharedPtr<CrossTensor>) -> String;
        /// Get the device where the tensor lives on
        fn device(tensor: &SharedPtr<CrossTensor>) -> Device;
        /// Get a string representation of a tensor
        fn repr(tensor: &SharedPtr<CrossTensor>) -> String;

        /// Add an empty dimension to a tensor at the given dimension.
        fn unsqueeze(
            tensor: &SharedPtr<CrossTensor>,
            dim: i64
        ) -> Result<SharedPtr<CrossTensor>>;

        // Tensor creation ops
        /// Create an empty tensor
        fn empty(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        /// Create a tensor filled with zeros
        fn zeros(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        /// Create a tensor filled with ones
        fn ones(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn full(
            dims: Vec<i64>,
            value: Scalar,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn rand(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn randn(
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn randint(
            low: i64,
            high: i64,
            dims: Vec<i64>,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn eye(
            n: i64,
            m: i64,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn arange(
            start: Scalar,
            end: Scalar,
            step: Scalar,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn linspace(
            start: Scalar,
            end: Scalar,
            steps: i64,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn logspace(
            start: Scalar,
            end: Scalar,
            steps: i64,
            base: Scalar,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;

        fn tensor(
            list: ScalarList,
            s_dtype: String,
            s_layout: String,
            s_device: Device,
            requires_grad: bool,
            pin_memory: bool,
            s_mem_fmt: String,
        ) -> Result<SharedPtr<CrossTensor>>;
    }
}

unsafe impl std::marker::Send for torch::CrossTensorRef {}
unsafe impl std::marker::Sync for torch::CrossTensorRef {}
