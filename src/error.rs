//! Error types and exit-code mapping.

use std::fmt;
use std::path::Path;

/// Top-level error kind, used to choose the process exit code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Invalid command line or option combination (exit code 2).
    Usage,
    /// Read, validation, conversion, strict-mode, or write failure (exit code 1).
    Conversion,
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl Error {
    pub fn usage(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Usage,
            message: message.into(),
            source: None,
        }
    }

    pub fn conversion(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Conversion,
            message: message.into(),
            source: None,
        }
    }

    pub fn with_source(
        kind: ErrorKind,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn path(path: &Path, message: impl fmt::Display) -> Self {
        Self::conversion(format!("{}: {message}", path.display()))
    }

    pub fn read(path: &Path, error: std::io::Error) -> Self {
        Self::with_source(
            ErrorKind::Conversion,
            format!("cannot read {}", path.display()),
            error,
        )
    }

    pub fn write(path: &Path, error: std::io::Error) -> Self {
        Self::with_source(
            ErrorKind::Conversion,
            format!("cannot write {}", path.display()),
            error,
        )
    }

    pub fn format(source_format: crate::io::Format, message: impl fmt::Display) -> Self {
        Self::conversion(format!("{source_format}: {message}"))
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn is_usage(&self) -> bool {
        self.kind == ErrorKind::Usage
    }

    pub fn exit_code(&self) -> i32 {
        match self.kind {
            ErrorKind::Usage => 2,
            ErrorKind::Conversion => 1,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)?;
        if let Some(source) = &self.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source.as_ref() as &(dyn std::error::Error + 'static))
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::with_source(ErrorKind::Conversion, "I/O error", error)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
