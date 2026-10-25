//! trine: a static triangle-mesh scene converter.
//!
//! The library exposes the canonical [`scene::Scene`] model, per-format readers
//! and writers through [`io`], and the end-to-end pipeline in [`convert`].

pub mod cli;
pub mod convert;
pub mod error;
pub mod io;
pub mod ops;
pub mod output;
pub mod report;
pub mod scene;

pub use error::{Error, Result};
