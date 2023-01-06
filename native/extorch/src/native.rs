extern crate cxx;

use rustler::NifUnitEnum;

/// Native access to libtorch (C++) functions and namespaces.
/// NOTE: Any change to this file will be overwitten.

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

/// DType enum.
// pub enum DType {
//     // Full name types
//     UInt8 = 0,
//     Int8 = 1,
//     Int16 = 2,
//     Int32 = 3,
//     Int64 = 4,
//     Float16 = 5,
//     Float32 = 6,
//     Float64 = 7,
//     BFloat16 = 8,
//     Complex32 = 9,
//     Complex64 = 10,
//     Complex128 = 11,
//     Bool = 12,
//     QUInt8 = 13,
//     QInt8 = 14,
//     QUInt2x4 = 15,
//     QUInt4x2 = 16,
//
//     // C-like type aliases
//     Byte = 0,
//     Char = 1,
//     Short = 2,
//     Int = 3,
//     Long = 4,
//     Half = 5,
//     Float = 6,
//     Double = 7,
//     CHalf = 9,
//     CFloat = 10,
//     CDouble = 11,
// }

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
        include!("extorch/include/wrapper.h");

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

/// Convert a tensor into a list
fn to_list(tensor: &SharedPtr<CrossTensor>) -> ScalarList;


        // Tensor operations
        
/// Add an empty dimension to a tensor at the given dimension.
fn unsqueeze(
    tensor: &SharedPtr<CrossTensor>,
    dim: i64
) -> Result<SharedPtr<CrossTensor>>;


        // Tensor creation operations.
        
/// Create an empty tensor.
fn empty(
    dims: Vec<i64>,
    s_dtype: String,
    s_layout: String,
    s_device: Device,
    requires_grad: bool,
    pin_memory: bool,
    s_mem_fmt: String,
) -> Result<SharedPtr<CrossTensor>>;

/// Create a tensor filled with zeros.
fn zeros(
    dims: Vec<i64>,
    s_dtype: String,
    s_layout: String,
    s_device: Device,
    requires_grad: bool,
    pin_memory: bool,
    s_mem_fmt: String,
) -> Result<SharedPtr<CrossTensor>>;

/// Create a tensor filled with ones.
fn ones(
    dims: Vec<i64>,
    s_dtype: String,
    s_layout: String,
    s_device: Device,
    requires_grad: bool,
    pin_memory: bool,
    s_mem_fmt: String,
) -> Result<SharedPtr<CrossTensor>>;

/// Create a tensor filled with a scalar value.
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

/// Create a tensor initialized with random values.
fn rand(
    dims: Vec<i64>,
    s_dtype: String,
    s_layout: String,
    s_device: Device,
    requires_grad: bool,
    pin_memory: bool,
    s_mem_fmt: String,
) -> Result<SharedPtr<CrossTensor>>;

/// Create a tensor initialized with random normal values.
fn randn(
    dims: Vec<i64>,
    s_dtype: String,
    s_layout: String,
    s_device: Device,
    requires_grad: bool,
    pin_memory: bool,
    s_mem_fmt: String,
) -> Result<SharedPtr<CrossTensor>>;

/// Create a tensor filled with random values ranging from `low` to `high`
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

/// Create a two-dimensional identity matrix of size nxm.
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

/// Create a 1D tensor containing values from `start` to `end` (non-inclusive)
/// spaced by `step`.
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

/// Create a 1D tensor containing a `steps` number of values between `start`
/// and `end`.
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

/// Create a 1D tensor containing a `steps` number of values between
/// `start` and `end`. The values are spaced on a logarithmic scale.
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

/// Create a n-dimensional tensor from a list of scalars.
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
