//! Bitwise operation functions for SPARQL extensions

use crate::model::{Literal, NamedNode, Term};
use crate::OxirsError;

/// BITAND - Bitwise AND operation on integers
pub(super) fn fn_bitand(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "BITAND requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(lit1), Term::Literal(lit2)) => {
            let num1 = lit1
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("BITAND requires integer arguments".to_string()))?;
            let num2 = lit2
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("BITAND requires integer arguments".to_string()))?;
            let result = num1 & num2;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "BITAND requires integer literals".to_string(),
        )),
    }
}

/// BITOR - Bitwise OR operation on integers
pub(super) fn fn_bitor(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "BITOR requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(lit1), Term::Literal(lit2)) => {
            let num1 = lit1
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("BITOR requires integer arguments".to_string()))?;
            let num2 = lit2
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("BITOR requires integer arguments".to_string()))?;
            let result = num1 | num2;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "BITOR requires integer literals".to_string(),
        )),
    }
}

/// BITXOR - Bitwise XOR operation on integers
pub(super) fn fn_bitxor(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "BITXOR requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(lit1), Term::Literal(lit2)) => {
            let num1 = lit1
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("BITXOR requires integer arguments".to_string()))?;
            let num2 = lit2
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("BITXOR requires integer arguments".to_string()))?;
            let result = num1 ^ num2;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "BITXOR requires integer literals".to_string(),
        )),
    }
}

/// BITNOT - Bitwise NOT operation on integer
pub(super) fn fn_bitnot(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "BITNOT requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let num = lit
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("BITNOT requires integer argument".to_string()))?;
            let result = !num;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "BITNOT requires integer literal".to_string(),
        )),
    }
}

/// LSHIFT - Left bit shift operation
pub(super) fn fn_lshift(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "LSHIFT requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(lit1), Term::Literal(lit2)) => {
            let num = lit1
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("LSHIFT requires integer value".to_string()))?;
            let shift = lit2.value().parse::<u32>().map_err(|_| {
                OxirsError::Query("LSHIFT requires non-negative integer shift".to_string())
            })?;
            // Prevent excessive shifts
            if shift > 63 {
                return Err(OxirsError::Query(
                    "LSHIFT shift amount must be <= 63".to_string(),
                ));
            }
            let result = num << shift;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "LSHIFT requires integer literals".to_string(),
        )),
    }
}

/// RSHIFT - Right bit shift operation (arithmetic shift)
pub(super) fn fn_rshift(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "RSHIFT requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(lit1), Term::Literal(lit2)) => {
            let num = lit1
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("RSHIFT requires integer value".to_string()))?;
            let shift = lit2.value().parse::<u32>().map_err(|_| {
                OxirsError::Query("RSHIFT requires non-negative integer shift".to_string())
            })?;
            // Prevent excessive shifts
            if shift > 63 {
                return Err(OxirsError::Query(
                    "RSHIFT shift amount must be <= 63".to_string(),
                ));
            }
            let result = num >> shift;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "RSHIFT requires integer literals".to_string(),
        )),
    }
}
