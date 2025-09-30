//! XSD Datatypes implementation for OxiRS
//! 
//! This module provides W3C XML Schema datatypes implementation compatible with SPARQL specifications.
//! Based on the Oxigraph oxsdatatypes library, adapted for OxiRS.

mod boolean;
mod decimal;
mod double;
mod duration;
mod float;
mod integer;
mod date_time;

pub use self::boolean::Boolean;
pub use self::decimal::{Decimal, ParseDecimalError, TooLargeForDecimalError};
pub use self::double::Double;
pub use self::duration::{
    DayTimeDuration, Duration, DurationOverflowError, OppositeSignInDurationComponentsError,
    ParseDurationError, YearMonthDuration,
};
pub use self::float::Float;
pub use self::integer::{Integer, TooLargeForIntegerError};
pub use self::date_time::{
    Date, DateTime, DateTimeOverflowError, GDay, GMonth, GMonthDay, GYear, GYearMonth,
    InvalidTimezoneError, ParseDateTimeError, Time, TimezoneOffset,
};

/// Error types for XSD datatype parsing and conversion
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XsdError {
    ParseError(String),
    OverflowError(String),
    InvalidValue(String),
}

impl std::fmt::Display for XsdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XsdError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            XsdError::OverflowError(msg) => write!(f, "Overflow error: {}", msg),
            XsdError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
        }
    }
}

impl std::error::Error for XsdError {}