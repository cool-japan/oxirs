//! # OWL 2 Profile Enumeration
//!
//! Defines the [`OwlProfile`] enum used by the unified OWL 2 reasoning dispatcher.
//! Each variant selects a different reasoning backend with different complexity
//! and expressivity guarantees.
//!
//! ## Profile semantics
//!
//! | Profile | Backend                                  | Complexity | Constructs |
//! |---------|------------------------------------------|------------|------------|
//! | `RL`    | [`crate::owl_rl::Owl2RlReasoner`]        | PTime data | RL fragment |
//! | `EL`    | [`crate::owl_el::Owl2ElReasoner`]        | PTime TBox | ⊓, ∃R.C, role chains |
//! | `QL`    | [`crate::owl_ql::QueryRewriter`]         | NLogSpace data | inv, ⊓, ∃R.⊤ |
//! | `RLEL`  | Combined RL + EL closure                 | PTime data | RL ∪ EL hybrid |
//! | `DL`    | [`crate::owl_dl::Owl2DLReasoner`]        | NExpTime   | full OWL 2 DL |
//!
//! ## Reference
//! <https://www.w3.org/TR/owl2-profiles/>

use std::fmt;

/// OWL 2 reasoning profile.
///
/// Selects the algorithm and supported language fragment used by
/// [`crate::owl2::reason_in_profile`] and the
/// [`crate::owl2::ProfileReasoner`] trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OwlProfile {
    /// OWL 2 RL — Rule Language profile.
    ///
    /// Forward-chaining materialisation using the W3C rule set
    /// (<https://www.w3.org/TR/owl2-profiles/#OWL_2_RL>).
    /// Polynomial in data size; complete for most ABox queries
    /// over the RL fragment.
    Rl,

    /// OWL 2 EL — Existential Language profile.
    ///
    /// Uses the EL completion algorithm (CR1–CR6 of the EL+ family,
    /// see Baader et al. "Pushing the EL Envelope", IJCAI 2005).
    /// Polynomial in TBox size; designed for very large terminologies
    /// such as SNOMED CT.
    El,

    /// OWL 2 QL — Query Language profile.
    ///
    /// Uses PerfectRef query rewriting
    /// (<https://www.w3.org/TR/owl2-profiles/#OWL_2_QL>).
    /// Compiles a conjunctive query into a Union of Conjunctive Queries
    /// (UCQ) that is sound and complete over any RDF data source.
    Ql,

    /// Combined OWL 2 RL ⊕ EL closure mode.
    ///
    /// Performs a fixed-point materialisation that runs the OWL 2 RL
    /// rule set, derives EL-shape axioms from the resulting closure,
    /// and re-runs EL classification, then loops until no new
    /// subsumptions appear.  Useful for hybrid TBoxes that exceed
    /// neither profile in isolation.
    RlEl,

    /// OWL 2 DL — full Description Logic profile (fallback).
    ///
    /// Delegates to the existing [`crate::owl_dl`] structural-subsumption
    /// engine.  Worst-case NExpTime — use only when no other profile fits.
    Dl,
}

impl OwlProfile {
    /// Return the canonical short name of the profile (`"RL"`, `"EL"`, `"QL"`,
    /// `"RLEL"`, `"DL"`).
    pub fn short_name(self) -> &'static str {
        match self {
            Self::Rl => "RL",
            Self::El => "EL",
            Self::Ql => "QL",
            Self::RlEl => "RLEL",
            Self::Dl => "DL",
        }
    }

    /// Parse a profile name from a string (case-insensitive).
    ///
    /// Accepted forms: `"RL"`, `"EL"`, `"QL"`, `"RLEL"`, `"RL+EL"`, `"DL"`.
    /// Returns `None` for unrecognised input.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_uppercase().as_str() {
            "RL" => Some(Self::Rl),
            "EL" => Some(Self::El),
            "QL" => Some(Self::Ql),
            "RLEL" | "RL+EL" | "EL+RL" | "ELRL" => Some(Self::RlEl),
            "DL" => Some(Self::Dl),
            _ => None,
        }
    }

    /// Return all profile variants in canonical order.
    pub fn all() -> &'static [Self] {
        &[Self::Rl, Self::El, Self::Ql, Self::RlEl, Self::Dl]
    }

    /// Indicate whether this profile is tractable (PTime in data) — used by
    /// callers that want to skip expensive `DL` reasoning.
    pub fn is_tractable(self) -> bool {
        matches!(self, Self::Rl | Self::El | Self::Ql | Self::RlEl)
    }
}

impl fmt::Display for OwlProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_short_names() {
        for profile in OwlProfile::all() {
            let parsed = OwlProfile::parse(profile.short_name())
                .expect("parsing canonical short name should succeed");
            assert_eq!(parsed, *profile);
        }
    }

    #[test]
    fn parse_aliases_and_case_insensitivity() {
        assert_eq!(OwlProfile::parse("rl"), Some(OwlProfile::Rl));
        assert_eq!(OwlProfile::parse(" RL+EL "), Some(OwlProfile::RlEl));
        assert_eq!(OwlProfile::parse("ELrl"), Some(OwlProfile::RlEl));
        assert_eq!(OwlProfile::parse("rlel"), Some(OwlProfile::RlEl));
        assert_eq!(OwlProfile::parse("dl"), Some(OwlProfile::Dl));
        assert_eq!(OwlProfile::parse("foo"), None);
    }

    #[test]
    fn tractability_classification() {
        assert!(OwlProfile::Rl.is_tractable());
        assert!(OwlProfile::El.is_tractable());
        assert!(OwlProfile::Ql.is_tractable());
        assert!(OwlProfile::RlEl.is_tractable());
        assert!(!OwlProfile::Dl.is_tractable());
    }

    #[test]
    fn display_matches_short_name() {
        assert_eq!(format!("{}", OwlProfile::Rl), "RL");
        assert_eq!(format!("{}", OwlProfile::RlEl), "RLEL");
    }
}
