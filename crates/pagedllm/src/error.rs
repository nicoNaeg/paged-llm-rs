//! The error type every fallible operation in this crate returns.

use std::fmt;
use std::path::PathBuf;

/// What can go wrong in the engine.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A tensor operation or a device call failed.
    Tensor(candle_core::Error),
    /// A file could not be read.
    Io(PathBuf, std::io::Error),
    /// A config file was not valid JSON, or was missing a field.
    Config(String),
    /// The config describes a model this implementation would have to
    /// misinterpret in order to run.
    Unsupported(String),
    /// A weight file does not carry a tensor the model needs, or carries it at
    /// the wrong shape.
    Weight(String),
}

/// A [`std::result::Result`] carrying this crate's error type.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(e) => write!(f, "tensor operation failed: {e}"),
            Self::Io(path, e) => write!(f, "reading {}: {e}", path.display()),
            Self::Config(what) => write!(f, "invalid config: {what}"),
            Self::Unsupported(what) => write!(f, "unsupported model: {what}"),
            Self::Weight(what) => write!(f, "weights: {what}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(e) => Some(e),
            Self::Io(_, e) => Some(e),
            Self::Config(_) | Self::Unsupported(_) | Self::Weight(_) => None,
        }
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Self::Tensor(e)
    }
}
