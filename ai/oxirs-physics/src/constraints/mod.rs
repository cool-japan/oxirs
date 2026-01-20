//! Physics Constraints and Validation

pub mod conservation_laws;
pub mod dimensional_analysis;
pub mod physical_bounds;

pub use conservation_laws::{ConservationChecker, ConservationLaw, ViolationReport};
pub use dimensional_analysis::{DimensionalAnalyzer, Dimensions, UnitChecker};
pub use physical_bounds::{BoundsValidator, PhysicalBounds};
