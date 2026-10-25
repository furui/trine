//! FBX support: robust import via `ufbx`, compact FBX 7.4 ASCII export.

mod read;
mod write;

pub use read::read;
pub use write::write;
