//! Warning collection, deduplication, and strict-mode accounting.

use std::collections::HashMap;
use std::fmt;

use crate::io::Format;

/// Category of a warning. Only [`WarningKind::is_data_loss`] categories fail
/// `--strict`; unit assumptions and GPU fallbacks never do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarningKind {
    /// A documented default was assumed (for example mm for STL input).
    Assumption,
    /// Input data cannot be represented by the target format.
    DroppedFeature,
    /// A referenced external asset could not be found.
    MissingAsset,
    /// A source entity was understood but not converted.
    UnsupportedEntity,
    /// A value was approximated lossily (for example PBR to Phong).
    Approximation,
    /// GPU acceleration was unavailable and the CPU path was used.
    GpuFallback,
    /// Nonstandard-but-valid behavior; informational.
    Note,
}

impl WarningKind {
    pub const fn is_data_loss(self) -> bool {
        matches!(
            self,
            WarningKind::DroppedFeature
                | WarningKind::MissingAsset
                | WarningKind::UnsupportedEntity
                | WarningKind::Approximation
        )
    }

    pub const fn label(self) -> &'static str {
        match self {
            WarningKind::Assumption => "assumption",
            WarningKind::DroppedFeature => "dropped",
            WarningKind::MissingAsset => "missing",
            WarningKind::UnsupportedEntity => "unsupported",
            WarningKind::Approximation => "approximated",
            WarningKind::GpuFallback => "gpu",
            WarningKind::Note => "note",
        }
    }
}

/// A single deduplicated warning.
#[derive(Debug, Clone)]
pub struct Warning {
    pub kind: WarningKind,
    pub code: String,
    pub target: Option<Format>,
    pub feature: String,
    pub count: usize,
}

impl Warning {
    pub fn message(&self) -> String {
        let mut text = format!("{}: {}", self.kind.label(), self.feature);
        if self.count > 1 {
            text.push_str(&format!(" [x{}]", self.count));
        }
        text
    }
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

/// Collects warnings and de-duplicates them by `(code, target, feature)`.
#[derive(Debug, Clone, Default)]
pub struct WarningSink {
    entries: Vec<Warning>,
    index: HashMap<(String, Option<Format>, String), usize>,
    quiet: bool,
}

impl WarningSink {
    pub fn new(quiet: bool) -> Self {
        Self {
            quiet,
            ..Self::default()
        }
    }

    pub fn set_quiet(&mut self, quiet: bool) {
        self.quiet = quiet;
    }

    pub fn is_quiet(&self) -> bool {
        self.quiet
    }

    pub fn push(&mut self, warning: Warning) {
        let key = (
            warning.code.clone(),
            warning.target,
            warning.feature.clone(),
        );
        if let Some(index) = self.index.get(&key) {
            self.entries[*index].count += warning.count;
            return;
        }
        self.index.insert(key, self.entries.len());
        self.entries.push(warning);
    }

    pub fn add(&mut self, kind: WarningKind, code: impl Into<String>, feature: impl Into<String>) {
        self.push(Warning {
            kind,
            code: code.into(),
            target: None,
            feature: feature.into(),
            count: 1,
        });
    }

    pub fn add_n(
        &mut self,
        kind: WarningKind,
        code: impl Into<String>,
        feature: impl Into<String>,
        count: usize,
    ) {
        if count == 0 {
            return;
        }
        self.push(Warning {
            kind,
            code: code.into(),
            target: None,
            feature: feature.into(),
            count,
        });
    }

    pub fn add_for(
        &mut self,
        format: Format,
        kind: WarningKind,
        code: impl Into<String>,
        feature: impl Into<String>,
        count: usize,
    ) {
        if count == 0 {
            return;
        }
        self.push(Warning {
            kind,
            code: code.into(),
            target: Some(format),
            feature: feature.into(),
            count,
        });
    }

    /// Emit all warnings to stderr unless quiet. Does not clear them.
    pub fn emit(&self) {
        if self.quiet {
            return;
        }
        for warning in &self.entries {
            eprintln!("warning: {}", warning.message());
        }
    }

    pub fn entries(&self) -> &[Warning] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of warnings that represent lost input data.
    pub fn data_loss_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|warning| warning.kind.is_data_loss())
            .count()
    }

    /// Strict mode: return an error if any warning represents lost input data.
    pub fn strict_error(&self) -> Option<crate::error::Error> {
        let losses = self.data_loss_count();
        if losses == 0 {
            return None;
        }
        Some(crate::error::Error::conversion(format!(
            "--strict: conversion would lose input data ({losses} warning categories); rerun without --strict to continue"
        )))
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
    }

    /// Merge warnings from a reader into this sink, preserving de-duplication.
    pub fn merge(&mut self, other: &WarningSink) {
        for warning in &other.entries {
            self.push(warning.clone());
        }
    }
}

/// Helper for reporting dropped feature counts with consistent wording.
pub fn dropped(sink: &mut WarningSink, format: Format, code: &str, feature: &str, count: usize) {
    if count == 0 {
        return;
    }
    let plural = if count == 1 { "" } else { "s" };
    sink.add_for(
        format,
        WarningKind::DroppedFeature,
        code,
        format!(
            "{} output does not support {feature}; dropped {count} {feature}{plural}",
            format.extension().to_ascii_uppercase()
        ),
        count,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_increments_counts() {
        let mut sink = WarningSink::new(true);
        sink.add_for(
            Format::Stl,
            WarningKind::DroppedFeature,
            "materials",
            "materials",
            2,
        );
        sink.add_for(
            Format::Stl,
            WarningKind::DroppedFeature,
            "materials",
            "materials",
            3,
        );
        assert_eq!(sink.entries().len(), 1);
        assert_eq!(sink.entries()[0].count, 5);
    }

    #[test]
    fn different_targets_are_not_merged() {
        let mut sink = WarningSink::new(true);
        sink.add_for(Format::Stl, WarningKind::DroppedFeature, "uv", "UVs", 1);
        sink.add_for(Format::Obj, WarningKind::DroppedFeature, "uv", "UVs", 1);
        assert_eq!(sink.entries().len(), 2);
    }

    #[test]
    fn assumptions_do_not_fail_strict() {
        let mut sink = WarningSink::new(true);
        sink.add(WarningKind::Assumption, "unit", "assuming millimeters");
        assert!(sink.strict_error().is_none());
    }

    #[test]
    fn dropped_features_fail_strict() {
        let mut sink = WarningSink::new(true);
        sink.add_for(Format::Stl, WarningKind::DroppedFeature, "uv", "UVs", 4);
        assert!(sink.strict_error().is_some());
    }
}
