use crate::native::torch;
use crate::shared_types::{AtomString, Reference, Size, TensorOptions, TensorStruct};

use cxx::SharedPtr;
use rustler::{Decoder, Encoder, NifResult, ResourceArc};

impl<'a> From<SharedPtr<torch::CrossTensor>> for TensorStruct<'a> {
    fn from(value: SharedPtr<torch::CrossTensor>) -> Self {
        let size = Size {
            size: torch::size(&value).to_vec(),
        };
        let device = torch::device(&value);
        let dtype_name: String = torch::dtype(&value);
        let dtype = AtomString { name: dtype_name };

        let wrapped_tensor = torch::CrossTensorRef {
            tensor: value,
        };

        let resource = ResourceArc::new(wrapped_tensor);
        let reference = Reference::new();
        TensorStruct {
            resource,
            reference,
            size,
            dtype,
            device,
        }
    }
}
