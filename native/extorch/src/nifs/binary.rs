use crate::native::torch;
use crate::shared_types::{AtomString, Size, TensorStruct};

use rustler::{Binary, Error, NifResult};

/// Create a tensor from raw binary data (copies the data into libtorch memory).
#[rustler::nif]
pub fn from_binary<'a>(data: Binary<'a>, shape: Size, dtype: AtomString) -> NifResult<TensorStruct<'a>> {
    let result = torch::from_binary(data.as_slice(), shape.size, dtype.name);
    match result {
        Ok(tensor) => Ok(tensor.into()),
        Err(err) => {
            let msg = err.what().to_owned();
            Err(Error::RaiseTerm(Box::new(msg)))
        }
    }
}
