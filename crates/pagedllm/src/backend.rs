//! Which device holds the tensors and runs the kernels.

use std::fmt;

use candle_core::Device;

use crate::Result;

/// The device the engine runs on.
///
/// Every kernel in this crate ships with a CPU implementation. It is the
/// reference the GPU version is checked against, and it is also the only path
/// CI can execute: `MTLCreateSystemDefaultDevice` returns nil inside GitHub's
/// macOS runners, so a hosted job can compile the Metal path but not dispatch
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Host memory. Correctness reference for every kernel, not a serving
    /// target.
    #[default]
    Cpu,
    /// Apple GPU through Metal.
    #[cfg(feature = "metal")]
    Metal,
}

impl Backend {
    /// The fastest backend this build was compiled for and this machine can
    /// reach, falling back to [`Backend::Cpu`].
    pub fn detect() -> Self {
        #[cfg(feature = "metal")]
        if Device::new_metal(0).is_ok() {
            return Self::Metal;
        }
        Self::Cpu
    }

    /// Open the candle device this backend names.
    pub fn device(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "metal")]
            Self::Metal => Ok(Device::new_metal(0)?),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            #[cfg(feature = "metal")]
            Self::Metal => f.write_str("metal"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Backend;

    #[test]
    fn the_cpu_backend_opens_on_every_machine() {
        assert!(Backend::Cpu.device().is_ok());
    }

    #[test]
    fn detection_never_names_a_device_it_cannot_open() {
        assert!(Backend::detect().device().is_ok());
    }
}
