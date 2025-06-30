//! Common types and definitions for molecular memory management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Helper function for serde default deserialization of Instant
fn default_instant() -> Instant {
    Instant::now()
}

/// Access level for operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccessLevel {
    Read,
    Write,
    Execute,
    Admin,
}

/// Chromosome segment for data partitioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromosomeSegment {
    /// Segment identifier
    pub id: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Data density
    pub density: f64,
    /// Genes in this segment
    pub genes: Vec<Gene>,
}

/// Gene representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gene {
    /// Gene identifier
    pub id: String,
    /// Promoter region
    pub promoter: PromoterRegion,
    /// Coding sequence
    pub coding_sequence: Vec<usize>,
    /// Exons
    pub exons: Vec<Exon>,
    /// Introns
    pub introns: Vec<Intron>,
}

/// Promoter region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromoterRegion {
    /// TATA box position
    pub tata_box: Option<usize>,
    /// Enhancers
    pub enhancers: Vec<Enhancer>,
    /// Silencers
    pub silencers: Vec<Silencer>,
}

/// Exon (coding region)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exon {
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Reading frame
    pub reading_frame: ReadingFrame,
}

/// Intron (non-coding region)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intron {
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Splice sites
    pub splice_sites: (SpliceSite, SpliceSite),
}

/// Reading frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadingFrame {
    Frame0,
    Frame1,
    Frame2,
}

/// Splice site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpliceSite {
    /// Position
    pub position: usize,
    /// Site type
    pub site_type: SpliceSiteType,
}

/// Splice site type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpliceSiteType {
    Donor,
    Acceptor,
}

/// Enhancer element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Enhancer {
    /// Position
    pub position: usize,
    /// Strength
    pub strength: f64,
    /// Target genes
    pub target_genes: Vec<String>,
}

/// Silencer element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Silencer {
    /// Position
    pub position: usize,
    /// Strength
    pub strength: f64,
    /// Target genes
    pub target_genes: Vec<String>,
}

/// Methylation pattern
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethylationPattern {
    /// CpG sites
    pub cpg_sites: Vec<CpGSite>,
    /// Methylation level
    pub methylation_level: f64,
    /// Pattern stability
    pub stability: f64,
}

/// CpG site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpGSite {
    /// Position
    #[serde(default)]
    pub position: usize,
    /// Methylation status
    #[serde(default)]
    pub methylated: bool,
    /// Methylation timestamp
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

impl Default for CpGSite {
    fn default() -> Self {
        Self {
            position: 0,
            methylated: false,
            timestamp: Instant::now(),
        }
    }
}

impl PartialEq for CpGSite {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position && self.methylated == other.methylated
        // Exclude timestamp from comparison as Instant doesn't implement PartialEq
    }
}

/// Histone modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoneModification {
    /// Modification type
    #[serde(default)]
    pub modification_type: ModificationType,
    /// Position
    #[serde(default)]
    pub position: usize,
    /// Intensity
    #[serde(default)]
    pub intensity: f64,
    /// Timestamp
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

impl Default for HistoneModification {
    fn default() -> Self {
        Self {
            modification_type: ModificationType::Acetylation,
            position: 0,
            intensity: 0.0,
            timestamp: Instant::now(),
        }
    }
}

impl PartialEq for HistoneModification {
    fn eq(&self, other: &Self) -> bool {
        self.modification_type == other.modification_type
            && self.position == other.position
            && self.intensity == other.intensity
        // Exclude timestamp from comparison as Instant doesn't implement PartialEq
    }
}

/// Modification type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModificationType {
    Acetylation,
    Methylation,
    Phosphorylation,
    Ubiquitination,
    Sumoylation,
    ADP_Ribosylation,
}

impl Default for ModificationType {
    fn default() -> Self {
        Self::Acetylation
    }
}

/// Modification status
#[derive(Debug, Clone)]
pub enum ModificationStatus {
    Modified,
    Unmodified,
    Partial(f64),
}

/// Regulatory function
#[derive(Debug, Clone)]
pub enum RegulatoryFunction {
    Loading,
    Maintenance,
    Removal,
    Modification,
}
