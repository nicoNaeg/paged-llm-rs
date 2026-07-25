//! Reading tensors out of a safetensors file, by name and at a checked shape.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::{Error, Result};

/// Every tensor of one checkpoint, resident on the target device.
///
/// Read through [`candle_core::safetensors::load`] rather than a memory map.
/// The mapped reader is `unsafe`, which this workspace denies, and the copy it
/// avoids is one this engine pays anyway: serving keeps every weight on the
/// device for the life of the process, so a lazy mapping would only defer the
/// same transfer to the first token.
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    cast: Option<DType>,
}

impl Weights {
    /// Load a checkpoint onto `device`.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let tensors = candle_core::safetensors::load(path, device)
            .map_err(|e| Error::Weight(format!("{}: {e}", path.display())))?;
        Ok(Self {
            tensors,
            cast: None,
        })
    }

    /// Convert every tensor to `dtype` as it is handed out.
    ///
    /// candle's CPU backend has no bf16 matmul, so a bf16 checkpoint can only
    /// run there upcast. That is also how the full-scale comparison isolates
    /// its two variables: f32 on the CPU says whether the implementation is
    /// right, and bf16 on Metal says what the serving dtype costs against it.
    #[must_use]
    pub fn cast_to(mut self, dtype: DType) -> Self {
        self.cast = Some(dtype);
        self
    }

    /// Fetch a tensor, refusing one whose shape is not what the caller expects.
    ///
    /// The shape check is what turns a checkpoint from another architecture
    /// into an error naming the tensor, instead of a matmul failure several
    /// layers away or, worse, a broadcast that happens to fit.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Weight(format!("no tensor named {name}")))?;
        if tensor.dims() != shape {
            return Err(Error::Weight(format!(
                "{name} has shape {:?}, expected {shape:?}",
                tensor.dims()
            )));
        }
        match self.cast {
            Some(dtype) if dtype != tensor.dtype() => Ok(tensor.to_dtype(dtype)?),
            _ => Ok(tensor.clone()),
        }
    }

    /// Fetch a tensor only if the checkpoint carries it.
    pub fn get_optional(&self, name: &str, shape: &[usize]) -> Result<Option<Tensor>> {
        if self.tensors.contains_key(name) {
            self.get(name, shape).map(Some)
        } else {
            Ok(None)
        }
    }

    /// The dtype tensors come out at, which is the cast if one was asked for
    /// and the stored dtype otherwise.
    pub fn dtype(&self, reference: &str) -> Result<DType> {
        if let Some(dtype) = self.cast {
            return Ok(dtype);
        }
        self.tensors
            .get(reference)
            .map(Tensor::dtype)
            .ok_or_else(|| Error::Weight(format!("no tensor named {reference}")))
    }

    /// How many tensors the checkpoint holds.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}
