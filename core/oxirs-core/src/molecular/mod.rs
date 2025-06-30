//! Molecular-Level Memory Management
//!
//! This module implements biomimetic memory management inspired by cellular
//! and molecular processes for ultra-efficient RDF data storage and processing.
//!
//! ## Module Organization
//!
//! - `dna_structures` - DNA-inspired data structures and nucleotide representations
//! - `replication` - DNA replication machinery (polymerase, helicase, ligase, etc.)
//! - `cellular_division` - Mitotic apparatus and cell division processes
//! - `regulatory` - Regulatory proteins and checkpoint systems
//! - `types` - Common types and error definitions
//!
//! This refactored module structure improves maintainability by breaking down
//! the original 2544-line file into focused, logical components while preserving
//! all original functionality.

use crate::error::OxirsResult;
use crate::model::{Term, Triple};

// Sub-modules
mod dna_structures;
mod replication;
mod cellular_division;
mod regulatory;
mod types;

// Re-export main types from sub-modules
pub use dna_structures::{DnaDataStructure, NucleotideData, SpecialMarker};
pub use replication::{ReplicationMachinery, DnaPolymerase, Helicase, Ligase, Primase, ProofreadingSystem};
pub use cellular_division::{CellularDivision, MitoticApparatus, SpindleApparatus, Centrosome, CellCycleState};
pub use regulatory::{RegulatoryProtein, CheckpointSystem, CheckpointResult};
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Test that all main types are accessible
        let _dna = DnaDataStructure::new();
        let _replication = ReplicationMachinery::new();
        let _division = CellularDivision::new();
        let _protein = RegulatoryProtein::new("test".to_string(), RegulatoryFunction::Loading);
        let _checkpoint = CheckpointSystem::new();
    }

    #[test]
    fn test_molecular_integration() {
        // Test integration between modules
        let mut division = CellularDivision::new();
        let dna = DnaDataStructure::new();
        
        division.add_dna_content(dna);
        assert!(matches!(division.current_state(), CellCycleState::G1));
    }
}