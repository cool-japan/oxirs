//! Numeric and mathematical functions for SPARQL 1.2

use crate::model::{Literal, NamedNode, Term};
use crate::OxirsError;

/// ABS - Absolute value
pub(super) fn fn_abs(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ABS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ABS requires numeric argument".to_string()))?;
            let result = value.abs();

            // Preserve datatype
            let dt = lit.datatype();
            if dt.as_str() == "http://www.w3.org/2001/XMLSchema#integer" {
                Ok(Term::Literal(Literal::new_typed(
                    (result as i64).to_string(),
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                        .expect("XSD integer datatype IRI should be valid"),
                )))
            } else {
                Ok(Term::Literal(Literal::new_typed(
                    result.to_string(),
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                        .expect("XSD double datatype IRI should be valid"),
                )))
            }
        }
        _ => Err(OxirsError::Query(
            "ABS requires numeric literal".to_string(),
        )),
    }
}

/// CEIL - Ceiling (round up)
pub(super) fn fn_ceil(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "CEIL requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("CEIL requires numeric argument".to_string()))?;
            let result = value.ceil() as i64;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("XSD integer datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "CEIL requires numeric literal".to_string(),
        )),
    }
}

/// FLOOR - Floor (round down)
pub(super) fn fn_floor(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "FLOOR requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("FLOOR requires numeric argument".to_string()))?;
            let result = value.floor() as i64;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("XSD integer datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "FLOOR requires numeric literal".to_string(),
        )),
    }
}

/// ROUND - Round to nearest integer
pub(super) fn fn_round(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ROUND requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ROUND requires numeric argument".to_string()))?;
            let result = value.round() as i64;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("XSD integer datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ROUND requires numeric literal".to_string(),
        )),
    }
}

/// RAND - Generate random number [0, 1)
pub(super) fn fn_rand(_args: &[Term]) -> Result<Term, OxirsError> {
    use scirs2_core::random::{Random, Rng};
    let mut random = Random::default();
    let value: f64 = random.random();
    Ok(Term::Literal(Literal::new_typed(
        value.to_string(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
            .expect("XSD double datatype IRI should be valid"),
    )))
}

/// SQRT - Square root
pub(super) fn fn_sqrt(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SQRT requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("SQRT requires numeric argument".to_string()))?;
            if value < 0.0 {
                return Err(OxirsError::Query("SQRT of negative number".to_string()));
            }
            let result = value.sqrt();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "SQRT requires numeric literal".to_string(),
        )),
    }
}

/// SIN - Sine
pub(super) fn fn_sin(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SIN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("SIN requires numeric argument".to_string()))?;
            let result = value.sin();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "SIN requires numeric literal".to_string(),
        )),
    }
}

/// COS - Cosine
pub(super) fn fn_cos(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "COS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("COS requires numeric argument".to_string()))?;
            let result = value.cos();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "COS requires numeric literal".to_string(),
        )),
    }
}

/// TAN - Tangent
pub(super) fn fn_tan(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TAN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("TAN requires numeric argument".to_string()))?;
            let result = value.tan();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "TAN requires numeric literal".to_string(),
        )),
    }
}

/// ASIN - Arcsine
pub(super) fn fn_asin(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ASIN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ASIN requires numeric argument".to_string()))?;
            if !(-1.0..=1.0).contains(&value) {
                return Err(OxirsError::Query(
                    "ASIN argument must be between -1 and 1".to_string(),
                ));
            }
            let result = value.asin();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ASIN requires numeric literal".to_string(),
        )),
    }
}

/// ACOS - Arccosine
pub(super) fn fn_acos(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ACOS requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ACOS requires numeric argument".to_string()))?;
            if !(-1.0..=1.0).contains(&value) {
                return Err(OxirsError::Query(
                    "ACOS argument must be between -1 and 1".to_string(),
                ));
            }
            let result = value.acos();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ACOS requires numeric literal".to_string(),
        )),
    }
}

/// ATAN - Arctangent
pub(super) fn fn_atan(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ATAN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN requires numeric argument".to_string()))?;
            let result = value.atan();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ATAN requires numeric literal".to_string(),
        )),
    }
}

/// ATAN2 - Two-argument arctangent
pub(super) fn fn_atan2(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "ATAN2 requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(y_lit), Term::Literal(x_lit)) => {
            let y = y_lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN2 requires numeric arguments".to_string()))?;
            let x = x_lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ATAN2 requires numeric arguments".to_string()))?;
            let result = y.atan2(x);
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ATAN2 requires numeric literals".to_string(),
        )),
    }
}

/// EXP - Exponential function (e^x)
pub(super) fn fn_exp(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "EXP requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("EXP requires numeric argument".to_string()))?;
            let result = value.exp();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "EXP requires numeric literal".to_string(),
        )),
    }
}

/// LOG - Natural logarithm
pub(super) fn fn_log(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LOG requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("LOG requires numeric argument".to_string()))?;
            if value <= 0.0 {
                return Err(OxirsError::Query("LOG of non-positive number".to_string()));
            }
            let result = value.ln();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "LOG requires numeric literal".to_string(),
        )),
    }
}

/// LOG10 - Base-10 logarithm
pub(super) fn fn_log10(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LOG10 requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("LOG10 requires numeric argument".to_string()))?;
            if value <= 0.0 {
                return Err(OxirsError::Query(
                    "LOG10 of non-positive number".to_string(),
                ));
            }
            let result = value.log10();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "LOG10 requires numeric literal".to_string(),
        )),
    }
}

/// POW - Power function
pub(super) fn fn_pow(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "POW requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(base_lit), Term::Literal(exp_lit)) => {
            let base = base_lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("POW requires numeric arguments".to_string()))?;
            let exp = exp_lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("POW requires numeric arguments".to_string()))?;
            let result = base.powf(exp);
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "POW requires numeric literals".to_string(),
        )),
    }
}

/// SINH - Hyperbolic sine
pub(super) fn fn_sinh(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SINH requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let num = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("SINH requires numeric argument".to_string()))?;
            let result = num.sinh();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "SINH requires numeric literal".to_string(),
        )),
    }
}

/// COSH - Hyperbolic cosine
pub(super) fn fn_cosh(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "COSH requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let num = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("COSH requires numeric argument".to_string()))?;
            let result = num.cosh();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "COSH requires numeric literal".to_string(),
        )),
    }
}

/// TANH - Hyperbolic tangent
pub(super) fn fn_tanh(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TANH requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let num = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("TANH requires numeric argument".to_string()))?;
            let result = num.tanh();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "TANH requires numeric literal".to_string(),
        )),
    }
}

/// ASINH - Inverse hyperbolic sine
pub(super) fn fn_asinh(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ASINH requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let num = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ASINH requires numeric argument".to_string()))?;
            let result = num.asinh();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ASINH requires numeric literal".to_string(),
        )),
    }
}

/// ACOSH - Inverse hyperbolic cosine
pub(super) fn fn_acosh(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ACOSH requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let num = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ACOSH requires numeric argument".to_string()))?;
            // acosh is only defined for x >= 1
            if num < 1.0 {
                return Err(OxirsError::Query(
                    "ACOSH requires argument >= 1".to_string(),
                ));
            }
            let result = num.acosh();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ACOSH requires numeric literal".to_string(),
        )),
    }
}

/// ATANH - Inverse hyperbolic tangent
pub(super) fn fn_atanh(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ATANH requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let num = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("ATANH requires numeric argument".to_string()))?;
            // atanh is only defined for |x| < 1
            if num <= -1.0 || num >= 1.0 {
                return Err(OxirsError::Query(
                    "ATANH requires argument in range (-1, 1)".to_string(),
                ));
            }
            let result = num.atanh();
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                    .expect("XSD double datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ATANH requires numeric literal".to_string(),
        )),
    }
}

/// PI - Return the mathematical constant π (pi)
pub(super) fn fn_pi(_args: &[Term]) -> Result<Term, OxirsError> {
    Ok(Term::Literal(Literal::new_typed(
        std::f64::consts::PI.to_string(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
            .expect("XSD double datatype IRI should be valid"),
    )))
}

/// E - Return the mathematical constant e (Euler's number)
pub(super) fn fn_e(_args: &[Term]) -> Result<Term, OxirsError> {
    Ok(Term::Literal(Literal::new_typed(
        std::f64::consts::E.to_string(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
            .expect("XSD double datatype IRI should be valid"),
    )))
}

/// TAU - Return the mathematical constant τ (tau = 2π)
pub(super) fn fn_tau(_args: &[Term]) -> Result<Term, OxirsError> {
    Ok(Term::Literal(Literal::new_typed(
        std::f64::consts::TAU.to_string(),
        NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
            .expect("XSD double datatype IRI should be valid"),
    )))
}

/// SIGN - Return the sign of a number (-1, 0, or 1)
pub(super) fn fn_sign(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "SIGN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("SIGN requires numeric argument".to_string()))?;

            let sign = if value > 0.0 {
                1
            } else if value < 0.0 {
                -1
            } else {
                0
            };

            Ok(Term::Literal(Literal::new_typed(
                sign.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("XSD integer datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "SIGN requires numeric literal".to_string(),
        )),
    }
}

/// MOD - Modulo operation
pub(super) fn fn_mod(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "MOD requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(a_lit), Term::Literal(b_lit)) => {
            let a = a_lit
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("MOD requires integer arguments".to_string()))?;
            let b = b_lit
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("MOD requires integer arguments".to_string()))?;

            if b == 0 {
                return Err(OxirsError::Query("MOD by zero".to_string()));
            }

            let result = a % b;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("XSD integer datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "MOD requires numeric literals".to_string(),
        )),
    }
}

/// TRUNC - Truncate to integer (towards zero)
pub(super) fn fn_trunc(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TRUNC requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit
                .value()
                .parse::<f64>()
                .map_err(|_| OxirsError::Query("TRUNC requires numeric argument".to_string()))?;
            let result = value.trunc() as i64;
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("XSD integer datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "TRUNC requires numeric literal".to_string(),
        )),
    }
}

/// GCD - Greatest Common Divisor using Euclidean algorithm
pub(super) fn fn_gcd(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "GCD requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(a_lit), Term::Literal(b_lit)) => {
            let mut a = a_lit
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("GCD requires integer arguments".to_string()))?
                .abs();
            let mut b = b_lit
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("GCD requires integer arguments".to_string()))?
                .abs();

            // Euclidean algorithm
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }

            Ok(Term::Literal(Literal::new_typed(
                a.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("XSD integer datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "GCD requires numeric literals".to_string(),
        )),
    }
}

/// LCM - Least Common Multiple
pub(super) fn fn_lcm(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "LCM requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(a_lit), Term::Literal(b_lit)) => {
            let a = a_lit
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("LCM requires integer arguments".to_string()))?
                .abs();
            let b = b_lit
                .value()
                .parse::<i64>()
                .map_err(|_| OxirsError::Query("LCM requires integer arguments".to_string()))?
                .abs();

            if a == 0 || b == 0 {
                return Ok(Term::Literal(Literal::new_typed(
                    "0",
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                        .expect("XSD integer datatype IRI should be valid"),
                )));
            }

            // Calculate GCD first
            let mut gcd_a = a;
            let mut gcd_b = b;
            while gcd_b != 0 {
                let temp = gcd_b;
                gcd_b = gcd_a % gcd_b;
                gcd_a = temp;
            }

            // LCM = (a * b) / GCD(a, b)
            let result = (a / gcd_a) * b;

            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("XSD integer datatype IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "LCM requires numeric literals".to_string(),
        )),
    }
}
