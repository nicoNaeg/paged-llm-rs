//! The error type every fallible operation in this crate returns.

use std::fmt;

/// What can go wrong in the engine.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A tensor operation or a device call failed.
    Tensor(candle_core::Error),
}

/// A [`std::result::Result`] carrying this crate's error type.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(e) => write!(f, "tensor operation failed: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(e) => Some(e),
        }
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Self::Tensor(e)
    }
}
