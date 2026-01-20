//! Built-in SPARQL Functions
//!
//! This module implements comprehensive SPARQL 1.1 built-in functions
//! as specified in the W3C recommendation.

use crate::extensions::ExtensionRegistry;
use anyhow::Result;

// Module declarations
pub mod aggregates;
pub mod datetime_functions;
pub mod hash_functions;
pub mod numeric_functions;
pub mod other_functions;
pub mod string_functions;
pub mod type_checking;

// Re-export all function structs for internal use

/// Register comprehensive SPARQL 1.1 built-in functions
pub fn register_builtin_functions(registry: &ExtensionRegistry) -> Result<()> {
    // String functions
    registry.register_function(string_functions::StrFunction)?;
    registry.register_function(string_functions::LangFunction)?;
    registry.register_function(string_functions::DatatypeFunction)?;
    registry.register_function(string_functions::ConcatFunction)?;
    registry.register_function(string_functions::SubstrFunction)?;
    registry.register_function(string_functions::StrlenFunction)?;
    registry.register_function(string_functions::UcaseFunction)?;
    registry.register_function(string_functions::LcaseFunction)?;
    registry.register_function(string_functions::ContainsFunction)?;
    registry.register_function(string_functions::StrStartsFunction)?;
    registry.register_function(string_functions::StrEndsFunction)?;
    registry.register_function(string_functions::ReplaceFunction)?;
    registry.register_function(string_functions::RegexFunction)?;
    registry.register_function(string_functions::EncodeForUriFunction)?;
    registry.register_function(string_functions::SubstringBeforeFunction)?;
    registry.register_function(string_functions::SubstringAfterFunction)?;

    // Numeric functions
    registry.register_function(numeric_functions::AbsFunction)?;
    registry.register_function(numeric_functions::CeilFunction)?;
    registry.register_function(numeric_functions::FloorFunction)?;
    registry.register_function(numeric_functions::RoundFunction)?;
    registry.register_function(numeric_functions::RandFunction)?;

    // Type checking functions
    registry.register_function(type_checking::BoundFunction)?;
    registry.register_function(type_checking::IsIriFunction)?;
    registry.register_function(type_checking::IsLiteralFunction)?;

    // Date/time functions
    registry.register_function(datetime_functions::NowFunction)?;
    registry.register_function(datetime_functions::YearFunction)?;
    registry.register_function(datetime_functions::MonthFunction)?;
    registry.register_function(datetime_functions::DayFunction)?;
    registry.register_function(datetime_functions::HoursFunction)?;
    registry.register_function(datetime_functions::MinutesFunction)?;
    registry.register_function(datetime_functions::SecondsFunction)?;
    registry.register_function(datetime_functions::TimezoneFunction)?;

    // Hash functions
    registry.register_function(hash_functions::Md5Function)?;
    registry.register_function(hash_functions::Sha1Function)?;
    registry.register_function(hash_functions::Sha256Function)?;
    registry.register_function(hash_functions::Sha384Function)?;
    registry.register_function(hash_functions::Sha512Function)?;

    // Other functions
    registry.register_function(other_functions::UuidFunction)?;
    registry.register_function(other_functions::StruuidFunction)?;
    registry.register_function(other_functions::IriFunction)?;
    registry.register_function(other_functions::BnodeFunction)?;
    registry.register_function(other_functions::CoalesceFunction)?;
    registry.register_function(other_functions::IfFunction)?;
    registry.register_function(other_functions::SametermFunction)?;
    registry.register_function(other_functions::LangMatchesFunction)?;

    // Aggregate functions
    registry.register_aggregate(aggregates::CountAggregate)?;
    registry.register_aggregate(aggregates::SumAggregate)?;
    registry.register_aggregate(aggregates::MinAggregate)?;
    registry.register_aggregate(aggregates::MaxAggregate)?;
    registry.register_aggregate(aggregates::AvgAggregate)?;
    registry.register_aggregate(aggregates::SampleAggregate)?;
    registry.register_aggregate(aggregates::GroupConcatAggregate)?;

    Ok(())
}
