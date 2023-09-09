use crate::native::torch;
use crate::shared_types::{AtomString, Reference, Size, TensorStruct};

use cxx::SharedPtr;
use rustler::ResourceArc;

impl<'a> From<SharedPtr<torch::CrossTensor>> for TensorStruct<'a> {
    fn from(value: SharedPtr<torch::CrossTensor>) -> Self {
        let size = Size {
            size: torch::size(&value).unwrap_or_else(|_| &[0]).to_vec(),
        };
        let device = torch::device(&value).unwrap();
        let dtype_name: String = torch::dtype(&value).unwrap();
        let dtype = AtomString { name: dtype_name };

        let wrapped_tensor = torch::CrossTensorRef { tensor: value };

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
