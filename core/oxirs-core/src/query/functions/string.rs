//! String manipulation functions for SPARQL 1.2

use crate::model::{Literal, NamedNode, Term};
use crate::OxirsError;
use regex::Regex;

/// CONCAT - Concatenate strings
pub(super) fn fn_concat(args: &[Term]) -> Result<Term, OxirsError> {
    let mut result = String::new();

    for arg in args {
        match arg {
            Term::Literal(lit) => result.push_str(lit.value()),
            Term::NamedNode(nn) => result.push_str(nn.as_str()),
            _ => {
                return Err(OxirsError::Query(
                    "CONCAT requires string arguments".to_string(),
                ))
            }
        }
    }

    Ok(Term::Literal(Literal::new(&result)))
}

/// STRLEN - Get string length
pub(super) fn fn_strlen(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "STRLEN requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let len = lit.value().chars().count() as i64;
            Ok(Term::Literal(Literal::new_typed(
                len.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "STRLEN requires string literal".to_string(),
        )),
    }
}

/// SUBSTR - Extract substring
pub(super) fn fn_substr(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query(
            "SUBSTR requires 2 or 3 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(start_lit)) => {
            let string = str_lit.value();
            let start = start_lit
                .value()
                .parse::<usize>()
                .map_err(|_| OxirsError::Query("Invalid start position".to_string()))?;

            let result = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(len_lit) => {
                        let len = len_lit
                            .value()
                            .parse::<usize>()
                            .map_err(|_| OxirsError::Query("Invalid length".to_string()))?;
                        string.chars().skip(start - 1).take(len).collect::<String>()
                    }
                    _ => return Err(OxirsError::Query("Length must be numeric".to_string())),
                }
            } else {
                string.chars().skip(start - 1).collect::<String>()
            };

            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query(
            "SUBSTR requires string and numeric arguments".to_string(),
        )),
    }
}

/// REPLACE - Replace substring using regex
pub(super) fn fn_replace(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 3 || args.len() > 4 {
        return Err(OxirsError::Query(
            "REPLACE requires 3 or 4 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1], &args[2]) {
        (Term::Literal(text), Term::Literal(pattern), Term::Literal(replacement)) => {
            let flags = if args.len() == 4 {
                match &args[3] {
                    Term::Literal(f) => f.value(),
                    _ => return Err(OxirsError::Query("Flags must be string".to_string())),
                }
            } else {
                ""
            };

            // Build regex with flags
            let regex_str = if flags.contains('i') {
                format!("(?i){}", pattern.value())
            } else {
                pattern.value().to_string()
            };

            let regex = Regex::new(&regex_str)
                .map_err(|e| OxirsError::Query(format!("Invalid regex: {e}")))?;

            let result = regex.replace_all(text.value(), replacement.value());
            Ok(Term::Literal(Literal::new(result.as_ref())))
        }
        _ => Err(OxirsError::Query(
            "REPLACE requires string arguments".to_string(),
        )),
    }
}

/// REGEX - Test if string matches regex pattern
pub(super) fn fn_regex(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query(
            "REGEX requires 2 or 3 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(pattern)) => {
            let flags = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(f) => f.value(),
                    _ => return Err(OxirsError::Query("Flags must be string".to_string())),
                }
            } else {
                ""
            };

            let regex_str = if flags.contains('i') {
                format!("(?i){}", pattern.value())
            } else {
                pattern.value().to_string()
            };

            let regex = Regex::new(&regex_str)
                .map_err(|e| OxirsError::Query(format!("Invalid regex: {e}")))?;

            let matches = regex.is_match(text.value());
            Ok(Term::Literal(Literal::new_typed(
                if matches { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "REGEX requires string arguments".to_string(),
        )),
    }
}

/// STRAFTER - Get substring after a delimiter
pub(super) fn fn_strafter(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRAFTER requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(after_lit)) => {
            let string = str_lit.value();
            let after = after_lit.value();

            if let Some(pos) = string.find(after) {
                let result = &string[pos + after.len()..];
                Ok(Term::Literal(Literal::new(result)))
            } else {
                Ok(Term::Literal(Literal::new("")))
            }
        }
        _ => Err(OxirsError::Query(
            "STRAFTER requires string arguments".to_string(),
        )),
    }
}

/// STRBEFORE - Get substring before a delimiter
pub(super) fn fn_strbefore(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRBEFORE requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(before_lit)) => {
            let string = str_lit.value();
            let before = before_lit.value();

            if let Some(pos) = string.find(before) {
                let result = &string[..pos];
                Ok(Term::Literal(Literal::new(result)))
            } else {
                Ok(Term::Literal(Literal::new("")))
            }
        }
        _ => Err(OxirsError::Query(
            "STRBEFORE requires string arguments".to_string(),
        )),
    }
}

/// STRSTARTS - Check if string starts with prefix
pub(super) fn fn_strstarts(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRSTARTS requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(prefix_lit)) => {
            let result = str_lit.value().starts_with(prefix_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "STRSTARTS requires string arguments".to_string(),
        )),
    }
}

/// STRENDS - Check if string ends with suffix
pub(super) fn fn_strends(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRENDS requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(suffix_lit)) => {
            let result = str_lit.value().ends_with(suffix_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "STRENDS requires string arguments".to_string(),
        )),
    }
}

/// CONTAINS - Check if string contains substring
pub(super) fn fn_contains(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "CONTAINS requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(str_lit), Term::Literal(substr_lit)) => {
            let result = str_lit.value().contains(substr_lit.value());
            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "CONTAINS requires string arguments".to_string(),
        )),
    }
}

/// ENCODE_FOR_URI - URL encode string
pub(super) fn fn_encode_for_uri(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ENCODE_FOR_URI requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let encoded = urlencoding::encode(lit.value());
            Ok(Term::Literal(Literal::new(encoded.as_ref())))
        }
        _ => Err(OxirsError::Query(
            "ENCODE_FOR_URI requires string argument".to_string(),
        )),
    }
}

/// UCASE - Convert to uppercase
pub(super) fn fn_ucase(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "UCASE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => Ok(Term::Literal(Literal::new(lit.value().to_uppercase()))),
        _ => Err(OxirsError::Query(
            "UCASE requires string argument".to_string(),
        )),
    }
}

/// LCASE - Convert to lowercase
pub(super) fn fn_lcase(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LCASE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => Ok(Term::Literal(Literal::new(lit.value().to_lowercase()))),
        _ => Err(OxirsError::Query(
            "LCASE requires string argument".to_string(),
        )),
    }
}

/// CONCAT_WS - Concatenate with separator
pub(super) fn fn_concat_ws(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 {
        return Err(OxirsError::Query(
            "CONCAT_WS requires at least 2 arguments (separator and at least one string)"
                .to_string(),
        ));
    }

    // First argument is the separator
    let separator = match &args[0] {
        Term::Literal(lit) => lit.value(),
        _ => {
            return Err(OxirsError::Query(
                "CONCAT_WS separator must be a string literal".to_string(),
            ))
        }
    };

    // Remaining arguments are strings to concatenate
    let strings: Result<Vec<&str>, OxirsError> = args[1..]
        .iter()
        .map(|arg| match arg {
            Term::Literal(lit) => Ok(lit.value()),
            Term::NamedNode(nn) => Ok(nn.as_str()),
            _ => Err(OxirsError::Query(
                "CONCAT_WS requires string arguments".to_string(),
            )),
        })
        .collect();

    let result = strings?.join(separator);
    Ok(Term::Literal(Literal::new(&result)))
}

/// SPLIT - Split string by delimiter
pub(super) fn fn_split(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "SPLIT requires exactly 2 arguments (string and delimiter)".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(delimiter)) => {
            let parts: Vec<&str> = text.value().split(delimiter.value()).collect();
            // Since SPARQL doesn't have native array return type, we return JSON array as string
            let result = format!(
                "[{}]",
                parts
                    .iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query(
            "SPLIT requires string arguments".to_string(),
        )),
    }
}

/// LPAD - Left pad string to specified length
pub(super) fn fn_lpad(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query(
            "LPAD requires 2 or 3 arguments (string, length, [padString])".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(length_lit)) => {
            let target_length = length_lit
                .value()
                .parse::<usize>()
                .map_err(|_| OxirsError::Query("LPAD length must be numeric".to_string()))?;

            let pad_string = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(pad) => pad.value(),
                    _ => {
                        return Err(OxirsError::Query(
                            "LPAD pad string must be a string literal".to_string(),
                        ))
                    }
                }
            } else {
                " " // Default to space
            };

            let text_value = text.value();
            let current_length = text_value.chars().count();

            let result = if current_length >= target_length {
                text_value.to_string()
            } else {
                let pad_length = target_length - current_length;
                let pad_chars: Vec<char> = pad_string.chars().collect();
                if pad_chars.is_empty() {
                    return Err(OxirsError::Query(
                        "LPAD pad string cannot be empty".to_string(),
                    ));
                }

                let mut padding = String::new();
                for i in 0..pad_length {
                    padding.push(pad_chars[i % pad_chars.len()]);
                }
                format!("{}{}", padding, text_value)
            };

            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query(
            "LPAD requires string and numeric arguments".to_string(),
        )),
    }
}

/// RPAD - Right pad string to specified length
pub(super) fn fn_rpad(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(OxirsError::Query(
            "RPAD requires 2 or 3 arguments (string, length, [padString])".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(length_lit)) => {
            let target_length = length_lit
                .value()
                .parse::<usize>()
                .map_err(|_| OxirsError::Query("RPAD length must be numeric".to_string()))?;

            let pad_string = if args.len() == 3 {
                match &args[2] {
                    Term::Literal(pad) => pad.value(),
                    _ => {
                        return Err(OxirsError::Query(
                            "RPAD pad string must be a string literal".to_string(),
                        ))
                    }
                }
            } else {
                " " // Default to space
            };

            let text_value = text.value();
            let current_length = text_value.chars().count();

            let result = if current_length >= target_length {
                text_value.to_string()
            } else {
                let pad_length = target_length - current_length;
                let pad_chars: Vec<char> = pad_string.chars().collect();
                if pad_chars.is_empty() {
                    return Err(OxirsError::Query(
                        "RPAD pad string cannot be empty".to_string(),
                    ));
                }

                let mut padding = String::new();
                for i in 0..pad_length {
                    padding.push(pad_chars[i % pad_chars.len()]);
                }
                format!("{}{}", text_value, padding)
            };

            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query(
            "RPAD requires string and numeric arguments".to_string(),
        )),
    }
}

/// TRIM - Remove leading and trailing whitespace
pub(super) fn fn_trim(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "TRIM requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let trimmed = lit.value().trim();
            Ok(Term::Literal(Literal::new(trimmed)))
        }
        _ => Err(OxirsError::Query(
            "TRIM requires string literal".to_string(),
        )),
    }
}

/// LTRIM - Remove leading whitespace
pub(super) fn fn_ltrim(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LTRIM requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let trimmed = lit.value().trim_start();
            Ok(Term::Literal(Literal::new(trimmed)))
        }
        _ => Err(OxirsError::Query(
            "LTRIM requires string literal".to_string(),
        )),
    }
}

/// RTRIM - Remove trailing whitespace
pub(super) fn fn_rtrim(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "RTRIM requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let trimmed = lit.value().trim_end();
            Ok(Term::Literal(Literal::new(trimmed)))
        }
        _ => Err(OxirsError::Query(
            "RTRIM requires string literal".to_string(),
        )),
    }
}

/// REVERSE - Reverse a string
pub(super) fn fn_reverse(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "REVERSE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let reversed: String = lit.value().chars().rev().collect();
            Ok(Term::Literal(Literal::new(&reversed)))
        }
        _ => Err(OxirsError::Query(
            "REVERSE requires string literal".to_string(),
        )),
    }
}

/// REPEAT - Repeat a string n times
pub(super) fn fn_repeat(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "REPEAT requires exactly 2 arguments (string, count)".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(text), Term::Literal(count_lit)) => {
            let count = count_lit
                .value()
                .parse::<usize>()
                .map_err(|_| OxirsError::Query("REPEAT count must be numeric".to_string()))?;

            if count > 10000 {
                return Err(OxirsError::Query(
                    "REPEAT count too large (maximum 10000)".to_string(),
                ));
            }

            let result = text.value().repeat(count);
            Ok(Term::Literal(Literal::new(&result)))
        }
        _ => Err(OxirsError::Query(
            "REPEAT requires string and numeric arguments".to_string(),
        )),
    }
}

/// CAPITALIZE - Capitalize first letter of each word
pub(super) fn fn_capitalize(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "CAPITALIZE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let capitalized: String = lit
                .value()
                .split_whitespace()
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    }
                })
                .collect::<Vec<String>>()
                .join(" ");
            Ok(Term::Literal(Literal::new(&capitalized)))
        }
        _ => Err(OxirsError::Query(
            "CAPITALIZE requires string literal".to_string(),
        )),
    }
}

/// ISALPHA - Check if string contains only alphabetic characters
pub(super) fn fn_isalpha(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ISALPHA requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value();
            let result = !value.is_empty() && value.chars().all(|c| c.is_alphabetic());
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ISALPHA requires string literal".to_string(),
        )),
    }
}

/// ISDIGIT - Check if string contains only numeric digits
pub(super) fn fn_isdigit(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ISDIGIT requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value();
            let result = !value.is_empty() && value.chars().all(|c| c.is_ascii_digit());
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ISDIGIT requires string literal".to_string(),
        )),
    }
}

/// ISALNUM - Check if string contains only alphanumeric characters
pub(super) fn fn_isalnum(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ISALNUM requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value();
            let result = !value.is_empty() && value.chars().all(|c| c.is_alphanumeric());
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ISALNUM requires string literal".to_string(),
        )),
    }
}

/// ISWHITESPACE - Check if string contains only whitespace
pub(super) fn fn_iswhitespace(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "ISWHITESPACE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value();
            let result = !value.is_empty() && value.chars().all(|c| c.is_whitespace());
            Ok(Term::Literal(Literal::new_typed(
                result.to_string(),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")
                    .expect("W3C XSD schema IRI should be valid"),
            )))
        }
        _ => Err(OxirsError::Query(
            "ISWHITESPACE requires string literal".to_string(),
        )),
    }
}
