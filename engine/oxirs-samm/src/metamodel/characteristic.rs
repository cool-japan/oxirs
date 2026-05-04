//! Characteristic Model Elements
//!
//! Characteristics describe the semantics of Property values.

use super::{ElementMetadata, ModelElement};
use serde::{Deserialize, Serialize};

/// A Characteristic in the SAMM meta model
///
/// Characteristics describe the semantics of a Property's value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Characteristic {
    /// Element metadata (URN, names, descriptions)
    pub metadata: ElementMetadata,

    /// The data type of this characteristic
    pub data_type: Option<String>,

    /// The kind/type of characteristic
    pub kind: CharacteristicKind,

    /// Constraints applied to this characteristic
    pub constraints: Vec<Constraint>,
}

/// Types of characteristics in SAMM
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CharacteristicKind {
    /// Basic characteristic
    Trait,

    /// Quantifiable characteristic (with unit)
    Quantifiable {
        /// Unit of measurement
        unit: String,
    },

    /// Measurement (Quantifiable with specific value)
    Measurement {
        /// Unit of measurement
        unit: String,
    },

    /// Enumeration of values
    Enumeration {
        /// Possible enumeration values
        values: Vec<String>,
    },

    /// State characteristic (enumeration representing states)
    State {
        /// Possible state values
        values: Vec<String>,
        /// Default state value
        default_value: Option<String>,
    },

    /// Duration characteristic
    Duration {
        /// Unit of time duration
        unit: String,
    },

    /// Collection of values
    Collection {
        /// Characteristic of collection elements
        element_characteristic: Option<Box<Characteristic>>,
    },

    /// List (ordered collection)
    List {
        /// Characteristic of list elements
        element_characteristic: Option<Box<Characteristic>>,
    },

    /// Set (unordered unique collection)
    Set {
        /// Characteristic of set elements
        element_characteristic: Option<Box<Characteristic>>,
    },

    /// Sorted set
    SortedSet {
        /// Characteristic of sorted set elements
        element_characteristic: Option<Box<Characteristic>>,
    },

    /// Time series
    TimeSeries {
        /// Characteristic of time series elements
        element_characteristic: Option<Box<Characteristic>>,
    },

    /// Code (string with encoding)
    Code,

    /// Either (one of two alternatives)
    Either {
        /// Left alternative characteristic
        left: Box<Characteristic>,
        /// Right alternative characteristic
        right: Box<Characteristic>,
    },

    /// Single entity
    SingleEntity {
        /// Entity type identifier
        entity_type: String,
    },

    /// Structured value
    StructuredValue {
        /// Rule for deconstructing the structured value
        deconstruction_rule: String,
        /// Element identifiers
        elements: Vec<String>,
    },
}

/// Constraints that can be applied to characteristics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Constraint {
    /// Language constraint
    LanguageConstraint {
        /// ISO 639 language code
        language_code: String,
    },

    /// Locale constraint
    LocaleConstraint {
        /// BCP 47 locale code
        locale_code: String,
    },

    /// Range constraint
    RangeConstraint {
        /// Minimum value (as string representation)
        min_value: Option<String>,
        /// Maximum value (as string representation)
        max_value: Option<String>,
        /// Lower bound inclusion definition
        lower_bound_definition: BoundDefinition,
        /// Upper bound inclusion definition
        upper_bound_definition: BoundDefinition,
    },

    /// Length constraint
    LengthConstraint {
        /// Minimum length
        min_value: Option<u64>,
        /// Maximum length
        max_value: Option<u64>,
    },

    /// Regular expression constraint
    RegularExpressionConstraint {
        /// Regular expression pattern
        pattern: String,
    },

    /// Encoding constraint
    EncodingConstraint {
        /// Character encoding (e.g., UTF-8, ASCII)
        encoding: String,
    },

    /// Fixed point constraint
    FixedPointConstraint {
        /// Number of integer digits
        integer: u32,
        /// Number of decimal digits
        scale: u32,
    },
}

/// Bound definition for range constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundDefinition {
    /// Open bound (exclusive)
    Open,
    /// Closed bound (inclusive)
    AtLeast,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
}

impl Characteristic {
    /// Create a new Characteristic
    pub fn new(urn: String, kind: CharacteristicKind) -> Self {
        Self {
            metadata: ElementMetadata::new(urn),
            data_type: None,
            kind,
            constraints: Vec::new(),
        }
    }

    /// Set the data type
    pub fn with_data_type(mut self, data_type: String) -> Self {
        self.data_type = Some(data_type);
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Get the kind of this characteristic
    pub fn kind(&self) -> &CharacteristicKind {
        &self.kind
    }
}

impl ModelElement for Characteristic {
    fn urn(&self) -> &str {
        &self.metadata.urn
    }

    fn metadata(&self) -> &ElementMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_characteristic_creation() {
        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#TestCharacteristic".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string());

        assert_eq!(characteristic.name(), "TestCharacteristic");
        assert_eq!(characteristic.data_type, Some("xsd:string".to_string()));
    }

    #[test]
    fn test_enumeration_characteristic() {
        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#StatusEnum".to_string(),
            CharacteristicKind::Enumeration {
                values: vec!["Active".to_string(), "Inactive".to_string()],
            },
        );

        assert_eq!(characteristic.name(), "StatusEnum");
        matches!(
            characteristic.kind(),
            CharacteristicKind::Enumeration { .. }
        );
    }

    #[test]
    fn test_measurement_characteristic() {
        let characteristic = Characteristic::new(
            "urn:samm:org.example:1.0.0#Speed".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:kilometrePerHour".to_string(),
            },
        );

        assert_eq!(characteristic.name(), "Speed");
        matches!(
            characteristic.kind(),
            CharacteristicKind::Measurement { .. }
        );
    }

    // ─── samm-c:Either tests ──────────────────────────────────────────────

    #[test]
    fn test_either_characteristic_creation() {
        let left = Characteristic::new(
            "urn:samm:org.example:1.0.0#LeftChar".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string());

        let right = Characteristic::new(
            "urn:samm:org.example:1.0.0#RightChar".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:metre".to_string(),
            },
        );

        let either = Characteristic::new(
            "urn:samm:org.example:1.0.0#EitherChar".to_string(),
            CharacteristicKind::Either {
                left: Box::new(left.clone()),
                right: Box::new(right.clone()),
            },
        );

        assert_eq!(either.name(), "EitherChar");
        assert!(matches!(either.kind(), CharacteristicKind::Either { .. }));

        // Verify branches are accessible and match the originals
        if let CharacteristicKind::Either {
            left: got_left,
            right: got_right,
        } = either.kind()
        {
            assert_eq!(got_left.urn(), left.urn());
            assert_eq!(got_right.urn(), right.urn());
        } else {
            panic!("Expected CharacteristicKind::Either");
        }
    }

    #[test]
    fn test_either_characteristic_left_right_differ() {
        // Order matters: Either(A, B) != Either(B, A)
        let char_a = Characteristic::new(
            "urn:samm:org.example:1.0.0#A".to_string(),
            CharacteristicKind::Trait,
        )
        .with_data_type("xsd:string".to_string());

        let char_b = Characteristic::new(
            "urn:samm:org.example:1.0.0#B".to_string(),
            CharacteristicKind::Measurement {
                unit: "unit:kilogram".to_string(),
            },
        );

        let either_ab = CharacteristicKind::Either {
            left: Box::new(char_a.clone()),
            right: Box::new(char_b.clone()),
        };
        let either_ba = CharacteristicKind::Either {
            left: Box::new(char_b.clone()),
            right: Box::new(char_a.clone()),
        };

        assert_ne!(
            either_ab, either_ba,
            "Either(A,B) must not equal Either(B,A)"
        );
        assert_eq!(
            either_ab,
            either_ab.clone(),
            "Either(A,B) must equal itself"
        );
    }

    #[test]
    fn test_either_nested_either() {
        // Either can nest: Either(Either(A, B), C) is valid
        let char_a = Characteristic::new(
            "urn:samm:org.example:1.0.0#A".to_string(),
            CharacteristicKind::Trait,
        );
        let char_b = Characteristic::new(
            "urn:samm:org.example:1.0.0#B".to_string(),
            CharacteristicKind::Code,
        );
        let char_c = Characteristic::new(
            "urn:samm:org.example:1.0.0#C".to_string(),
            CharacteristicKind::Enumeration {
                values: vec!["X".to_string(), "Y".to_string()],
            },
        );

        let inner_either = Characteristic::new(
            "urn:samm:org.example:1.0.0#InnerEither".to_string(),
            CharacteristicKind::Either {
                left: Box::new(char_a),
                right: Box::new(char_b),
            },
        );

        let outer_either = Characteristic::new(
            "urn:samm:org.example:1.0.0#OuterEither".to_string(),
            CharacteristicKind::Either {
                left: Box::new(inner_either.clone()),
                right: Box::new(char_c),
            },
        );

        assert!(matches!(
            outer_either.kind(),
            CharacteristicKind::Either { .. }
        ));

        // Verify inner nesting
        if let CharacteristicKind::Either { left, .. } = outer_either.kind() {
            assert_eq!(left.urn(), inner_either.urn());
            assert!(matches!(left.kind(), CharacteristicKind::Either { .. }));
        } else {
            panic!("Expected outer CharacteristicKind::Either");
        }
    }

    #[test]
    fn test_either_round_trips_through_kind_ref() {
        // Confirm that kind() returns a reference that correctly identifies Either
        let left = Characteristic::new(
            "urn:samm:org.example:1.0.0#L".to_string(),
            CharacteristicKind::Trait,
        );
        let right = Characteristic::new(
            "urn:samm:org.example:1.0.0#R".to_string(),
            CharacteristicKind::Trait,
        );
        let either = Characteristic::new(
            "urn:samm:org.example:1.0.0#E".to_string(),
            CharacteristicKind::Either {
                left: Box::new(left),
                right: Box::new(right),
            },
        );

        // kind() ref matches Either pattern
        let is_either = matches!(either.kind(), CharacteristicKind::Either { .. });
        assert!(is_either);

        // Serializes to the right name fragment
        let urn_fragment = either.urn().split('#').next_back().unwrap_or("");
        assert_eq!(urn_fragment, "E");
    }
}
