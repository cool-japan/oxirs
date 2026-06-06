//! Built-in SPARQL Functions — Facade
//!
//! This module re-exports all SPARQL 1.1 built-in functions and provides the
//! central registration entry point. Implementation details live in:
//!
//! - [`crate::builtin_string`]   — string, type-check, logical, and regex functions
//! - [`crate::builtin_numeric`]  — numeric/math functions
//! - [`crate::builtin_datetime`] — date/time, hash, and aggregate functions

use crate::builtin_datetime::{
    AvgAggregate, CountAggregate, DayFunction, GroupConcatAggregate, HoursFunction, MaxAggregate,
    Md5Function, MinAggregate, MinutesFunction, MonthFunction, NowFunction, SampleAggregate,
    SecondsFunction, Sha1Function, Sha256Function, Sha384Function, Sha512Function, SumAggregate,
    TimezoneFunction, TzFunction, YearFunction,
};
use crate::builtin_numeric::{
    AbsFunction, CeilFunction, FloorFunction, RandFunction, RoundFunction,
};
use crate::builtin_string::{
    BlankFunction, BoundFunction, CoalesceFunction, ConcatFunction, ContainsFunction,
    DatatypeFunction, EncodeForUriFunction, IfFunction, IriFunction, IsBlankFunction,
    IsIriFunction, IsLiteralFunction, IsNumericFunction, IsUriFunction, LangFunction,
    LcaseFunction, LiteralFunction, RegexFunction, ReplaceFunction, StrFunction, StrendsFunction,
    StrlenFunction, StrstartsFunction, SubstrFunction, UcaseFunction, UriFunction,
};
use crate::extensions::ExtensionRegistry;
use anyhow::Result;

/// Register all built-in SPARQL functions and aggregates into the given registry.
pub fn register_builtin_functions(registry: &ExtensionRegistry) -> Result<()> {
    // ── String / RDF-term predicates ─────────────────────────────────────────
    registry.register_function(StrFunction)?;
    registry.register_function(LangFunction)?;
    registry.register_function(DatatypeFunction)?;
    registry.register_function(BoundFunction)?;
    registry.register_function(IriFunction)?;
    registry.register_function(UriFunction)?;
    registry.register_function(BlankFunction)?;
    registry.register_function(LiteralFunction)?;

    // ── String manipulation ───────────────────────────────────────────────────
    registry.register_function(StrlenFunction)?;
    registry.register_function(SubstrFunction)?;
    registry.register_function(UcaseFunction)?;
    registry.register_function(LcaseFunction)?;
    registry.register_function(StrstartsFunction)?;
    registry.register_function(StrendsFunction)?;
    registry.register_function(ContainsFunction)?;
    registry.register_function(ConcatFunction)?;
    registry.register_function(EncodeForUriFunction)?;
    registry.register_function(ReplaceFunction)?;

    // ── Numeric / math ────────────────────────────────────────────────────────
    registry.register_function(AbsFunction)?;
    registry.register_function(RoundFunction)?;
    registry.register_function(CeilFunction)?;
    registry.register_function(FloorFunction)?;
    registry.register_function(RandFunction)?;

    // ── Date / time ───────────────────────────────────────────────────────────
    registry.register_function(NowFunction)?;
    registry.register_function(YearFunction)?;
    registry.register_function(MonthFunction)?;
    registry.register_function(DayFunction)?;
    registry.register_function(HoursFunction)?;
    registry.register_function(MinutesFunction)?;
    registry.register_function(SecondsFunction)?;
    registry.register_function(TimezoneFunction)?;
    registry.register_function(TzFunction)?;

    // ── Hash functions ────────────────────────────────────────────────────────
    registry.register_function(Md5Function)?;
    registry.register_function(Sha1Function)?;
    registry.register_function(Sha256Function)?;
    registry.register_function(Sha384Function)?;
    registry.register_function(Sha512Function)?;

    // ── Type checking ─────────────────────────────────────────────────────────
    registry.register_function(IsIriFunction)?;
    registry.register_function(IsUriFunction)?;
    registry.register_function(IsBlankFunction)?;
    registry.register_function(IsLiteralFunction)?;
    registry.register_function(IsNumericFunction)?;

    // ── Logical ───────────────────────────────────────────────────────────────
    registry.register_function(IfFunction)?;
    registry.register_function(CoalesceFunction)?;

    // ── Regex ─────────────────────────────────────────────────────────────────
    registry.register_function(RegexFunction)?;

    // ── RDF-star TRIPLE functions ─────────────────────────────────────────────
    registry.register_function(crate::triple_functions::TripleFunction)?;
    registry.register_function(crate::triple_functions::SubjectFunction)?;
    registry.register_function(crate::triple_functions::PredicateFunction)?;
    registry.register_function(crate::triple_functions::ObjectFunction)?;
    registry.register_function(crate::triple_functions::IsTripleFunction)?;

    // ── Enhanced string functions (SPARQL 1.1+) ───────────────────────────────
    registry.register_function(crate::string_functions_ext::StrBeforeFunction)?;
    registry.register_function(crate::string_functions_ext::StrAfterFunction)?;
    registry.register_function(crate::string_functions_ext::StrLangFunction)?;
    registry.register_function(crate::string_functions_ext::StrLangDirFunction)?;
    registry.register_function(crate::string_functions_ext::StrDtFunction)?;

    // ── Aggregates ────────────────────────────────────────────────────────────
    registry.register_aggregate(CountAggregate)?;
    registry.register_aggregate(SumAggregate)?;
    registry.register_aggregate(AvgAggregate)?;
    registry.register_aggregate(MinAggregate)?;
    registry.register_aggregate(MaxAggregate)?;
    registry.register_aggregate(GroupConcatAggregate)?;
    registry.register_aggregate(SampleAggregate)?;

    Ok(())
}
