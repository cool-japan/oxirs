//! Date/time functions for SPARQL 1.2

use crate::model::{Literal, NamedNode, Term};
use crate::OxirsError;
use chrono::{DateTime, Datelike, Timelike, Utc};

/// NOW - Get current date/time
pub(super) fn fn_now(_args: &[Term]) -> Result<Term, OxirsError> {
    let now = Utc::now();
    Ok(Term::Literal(Literal::new_typed(
        now.to_rfc3339(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#dateTime")
            .expect("W3C XSD schema IRI should be valid"),
    )))
}

/// YEAR - Extract year from date/time
pub(super) fn fn_year(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "YEAR requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.year().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "YEAR requires dateTime literal".to_string(),
        )),
    }
}

/// MONTH - Extract month from date/time
pub(super) fn fn_month(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "MONTH requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.month().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "MONTH requires dateTime literal".to_string(),
        )),
    }
}

/// DAY - Extract day from date/time
pub(super) fn fn_day(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "DAY requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.day().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "DAY requires dateTime literal".to_string(),
        )),
    }
}

/// HOURS - Extract hours from date/time
pub(super) fn fn_hours(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "HOURS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.hour().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "HOURS requires dateTime literal".to_string(),
        )),
    }
}

/// MINUTES - Extract minutes from date/time
pub(super) fn fn_minutes(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "MINUTES requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                dt.minute().to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "MINUTES requires dateTime literal".to_string(),
        )),
    }
}

/// SECONDS - Extract seconds from date/time
pub(super) fn fn_seconds(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SECONDS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            Ok(Term::Literal(Literal::new_typed(
                format!("{}.{:09}", dt.second(), dt.nanosecond()),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "SECONDS requires dateTime literal".to_string(),
        )),
    }
}

/// TIMEZONE - Extract timezone as duration
pub(super) fn fn_timezone(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TIMEZONE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            let offset = dt.offset();
            let hours = offset.local_minus_utc() / 3600;
            let minutes = (offset.local_minus_utc() % 3600) / 60;

            let duration = if hours == 0 && minutes == 0 {
                "PT0S".to_string()
            } else {
                format!("PT{}H{}M", hours.abs(), minutes.abs())
            };

            Ok(Term::Literal(Literal::new_typed(
                &duration,
                NamedNode::new("http://www.w3.org/2001/XMLSchema#dayTimeDuration")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "TIMEZONE requires dateTime literal".to_string(),
        )),
    }
}

/// TZ - Extract timezone as string
pub(super) fn fn_tz(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TZ requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = DateTime::parse_from_rfc3339(lit.value())
                .map_err(|_| OxirsError::Query("Invalid dateTime".to_string()))?;
            let offset = dt.offset();
            let hours = offset.local_minus_utc() / 3600;
            let minutes = (offset.local_minus_utc() % 3600) / 60;

            let tz_string = if hours == 0 && minutes == 0 {
                "Z".to_string()
            } else {
                format!("{:+03}:{:02}", hours, minutes.abs())
            };

            Ok(Term::Literal(Literal::new(&tz_string)))
        }
        _ => Err(OxirsError::Query(
            "TZ requires dateTime literal".to_string(),
        )),
    }
}

/// ADJUST - Adjust timezone (placeholder implementation)
#[allow(dead_code)]
pub(super) fn fn_adjust(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "ADJUST requires exactly 2 arguments".to_string(),
        ));
    }

    // For now, just return the first argument (datetime value)
    // A full implementation would apply timezone adjustment
    Ok(args[0].clone())
}
