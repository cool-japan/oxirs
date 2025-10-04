//! SWRL (Semantic Web Rule Language) - Vocabulary
//!
//! This module implements SWRL rule components.

/// SWRL vocabulary constants
pub mod vocabulary {
    // SWRL Core
    pub const SWRL_IMPLIES: &str = "http://www.w3.org/2003/11/swrl#Implies";
    pub const SWRL_BODY: &str = "http://www.w3.org/2003/11/swrl#body";
    pub const SWRL_HEAD: &str = "http://www.w3.org/2003/11/swrl#head";

    // SWRL Atoms
    pub const SWRL_CLASS_ATOM: &str = "http://www.w3.org/2003/11/swrl#ClassAtom";
    pub const SWRL_INDIVIDUAL_PROPERTY_ATOM: &str =
        "http://www.w3.org/2003/11/swrl#IndividualPropertyAtom";
    pub const SWRL_DATAVALUE_PROPERTY_ATOM: &str =
        "http://www.w3.org/2003/11/swrl#DatavaluedPropertyAtom";
    pub const SWRL_BUILTIN_ATOM: &str = "http://www.w3.org/2003/11/swrl#BuiltinAtom";
    pub const SWRL_SAME_INDIVIDUAL_ATOM: &str = "http://www.w3.org/2003/11/swrl#SameIndividualAtom";
    pub const SWRL_DIFFERENT_INDIVIDUALS_ATOM: &str =
        "http://www.w3.org/2003/11/swrl#DifferentIndividualsAtom";

    // SWRL Atom Properties
    pub const SWRL_CLASS_PREDICATE: &str = "http://www.w3.org/2003/11/swrl#classPredicate";
    pub const SWRL_PROPERTY_PREDICATE: &str = "http://www.w3.org/2003/11/swrl#propertyPredicate";
    pub const SWRL_BUILTIN: &str = "http://www.w3.org/2003/11/swrl#builtin";
    pub const SWRL_ARGUMENT1: &str = "http://www.w3.org/2003/11/swrl#argument1";
    pub const SWRL_ARGUMENT2: &str = "http://www.w3.org/2003/11/swrl#argument2";
    pub const SWRL_ARGUMENTS: &str = "http://www.w3.org/2003/11/swrl#arguments";

    // SWRL Variables and Literals
    pub const SWRL_VARIABLE: &str = "http://www.w3.org/2003/11/swrl#Variable";
    pub const SWRL_INDIVIDUAL: &str = "http://www.w3.org/2003/11/swrl#Individual";

    // Built-in namespaces
    pub const SWRLB_NS: &str = "http://www.w3.org/2003/11/swrlb#";
    pub const SWRLM_NS: &str = "http://www.w3.org/2003/11/swrlm#";
    pub const SWRLT_NS: &str = "http://www.w3.org/2003/11/swrlt#";
    pub const SWRLX_NS: &str = "http://www.w3.org/2003/11/swrlx#";

    // SWRL-X Temporal Extensions
    pub const SWRLX_TEMPORAL_NS: &str = "http://www.w3.org/2003/11/swrlx/temporal#";

    // Temporal predicates
    pub const SWRLX_BEFORE: &str = "http://www.w3.org/2003/11/swrlx/temporal#before";
    pub const SWRLX_AFTER: &str = "http://www.w3.org/2003/11/swrlx/temporal#after";
    pub const SWRLX_DURING: &str = "http://www.w3.org/2003/11/swrlx/temporal#during";
    pub const SWRLX_OVERLAPS: &str = "http://www.w3.org/2003/11/swrlx/temporal#overlaps";
    pub const SWRLX_MEETS: &str = "http://www.w3.org/2003/11/swrlx/temporal#meets";
    pub const SWRLX_STARTS: &str = "http://www.w3.org/2003/11/swrlx/temporal#starts";
    pub const SWRLX_FINISHES: &str = "http://www.w3.org/2003/11/swrlx/temporal#finishes";
    pub const SWRLX_EQUALS: &str = "http://www.w3.org/2003/11/swrlx/temporal#equals";

    // Interval operations
    pub const SWRLX_INTERVAL_DURATION: &str =
        "http://www.w3.org/2003/11/swrlx/temporal#intervalDuration";
    pub const SWRLX_INTERVAL_START: &str = "http://www.w3.org/2003/11/swrlx/temporal#intervalStart";
    pub const SWRLX_INTERVAL_END: &str = "http://www.w3.org/2003/11/swrlx/temporal#intervalEnd";
}
