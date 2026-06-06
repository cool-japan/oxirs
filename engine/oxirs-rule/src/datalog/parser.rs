//! Datalog parser
//!
//! Parses Datalog programs from text. Supports:
//! - Facts: `parent(alice, bob).`
//! - Rules: `ancestor(X, Y) :- parent(X, Y).`
//! - Recursive rules: `ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).`
//! - Comments: `% comment` (rest of line is ignored)
//! - Integer constants: bare digit sequences
//! - Boolean constants: `true` / `false`
//! - String constants: lowercase identifiers or double-quoted strings
//! - Variables: identifiers starting with uppercase or `_`

use super::{
    DatalogAtom, DatalogError, DatalogFact, DatalogProgram, DatalogRule, DatalogTerm, DatalogValue,
};

/// Parse an entire Datalog program (facts and rules) from text.
pub fn parse_program(input: &str) -> Result<DatalogProgram, DatalogError> {
    let mut program = DatalogProgram::new();

    // Remove comments (% to end of line) and split on `.`
    // We need to be careful: quoted strings might contain `.`
    let cleaned = strip_comments(input);

    for clause in split_clauses(&cleaned) {
        let clause = clause.trim();
        if clause.is_empty() {
            continue;
        }

        if clause.contains(":-") {
            // It's a rule
            let rule = parse_rule(clause)?;
            program.add_rule(rule);
        } else {
            // It's a fact
            let fact = parse_fact_str(clause)?;
            program.add_fact(fact);
        }
    }

    Ok(program)
}

/// Parse a single rule string: `head :- body1, body2, ...`
/// The trailing `.` is optional.
pub fn parse_rule(input: &str) -> Result<DatalogRule, DatalogError> {
    let input = input.trim().trim_end_matches('.');
    let parts: Vec<&str> = input.splitn(2, ":-").collect();

    if parts.len() != 2 {
        return Err(DatalogError::ParseError(format!(
            "expected ':-' in rule: {input}"
        )));
    }

    let head = parse_atom(parts[0].trim())?;
    let body_str = parts[1].trim();

    let body = if body_str.is_empty() {
        Vec::new()
    } else {
        split_atoms(body_str)
            .iter()
            .map(|s| parse_atom(s.trim()))
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok(DatalogRule { head, body })
}

/// Parse a single atom: `predicate(term1, term2, ...)`.
/// Also accepts nullary predicates: `predicate`.
pub fn parse_atom(input: &str) -> Result<DatalogAtom, DatalogError> {
    let input = input.trim();

    if let Some(paren_pos) = input.find('(') {
        // Atom with arguments
        let predicate = input[..paren_pos].trim().to_string();
        validate_predicate(&predicate)?;

        let rest = &input[paren_pos + 1..];
        let close_pos = rest.rfind(')').ok_or_else(|| {
            DatalogError::ParseError(format!("missing closing ')' in atom: {input}"))
        })?;

        let args_str = &rest[..close_pos];
        let terms = if args_str.trim().is_empty() {
            Vec::new()
        } else {
            split_terms(args_str)
                .iter()
                .map(|s| parse_term(s.trim()))
                .collect::<Result<Vec<_>, _>>()?
        };

        Ok(DatalogAtom { predicate, terms })
    } else {
        // Nullary atom (no parentheses)
        let predicate = input.to_string();
        validate_predicate(&predicate)?;
        Ok(DatalogAtom {
            predicate,
            terms: Vec::new(),
        })
    }
}

// ---------- internal helpers ----------

/// Strip `% comment` lines (from `%` to end of line).
fn strip_comments(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            if let Some(pos) = line.find('%') {
                // Check it's not inside a quoted string (simplified: assume no % in strings)
                &line[..pos]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Split text into clauses on `.` boundaries (respecting quoted strings and nested parens).
fn split_clauses(input: &str) -> Vec<String> {
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut paren_depth: usize = 0;

    for ch in input.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            '(' if !in_quotes => {
                paren_depth += 1;
                current.push(ch);
            }
            ')' if !in_quotes => {
                paren_depth = paren_depth.saturating_sub(1);
                current.push(ch);
            }
            '.' if !in_quotes && paren_depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    clauses.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        clauses.push(trimmed);
    }

    clauses
}

/// Split a comma-separated list of atoms (respecting nested parens).
fn split_atoms(input: &str) -> Vec<String> {
    split_on_comma_with_depth(input)
}

/// Split a comma-separated list of terms (respecting nested parens and quotes).
fn split_terms(input: &str) -> Vec<String> {
    split_on_comma_with_depth(input)
}

fn split_on_comma_with_depth(input: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut current = String::new();
    let mut depth: usize = 0;
    let mut in_quotes = false;

    for ch in input.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current.push(ch);
            }
            '(' if !in_quotes => {
                depth += 1;
                current.push(ch);
            }
            ')' if !in_quotes => {
                depth = depth.saturating_sub(1);
                current.push(ch);
            }
            ',' if !in_quotes && depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    items.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        items.push(trimmed);
    }

    items
}

/// Parse a fact string (atom that must be ground).
fn parse_fact_str(input: &str) -> Result<DatalogFact, DatalogError> {
    let atom = parse_atom(input)?;

    // All terms must be constants
    let mut args = Vec::new();
    for term in &atom.terms {
        match term {
            DatalogTerm::Constant(v) => args.push(v.clone()),
            DatalogTerm::Variable(v) => {
                return Err(DatalogError::ParseError(format!(
                    "facts cannot contain variables, found variable '{v}' in: {input}"
                )));
            }
        }
    }

    Ok(DatalogFact {
        predicate: atom.predicate,
        args,
    })
}

/// Parse a single term string.
fn parse_term(input: &str) -> Result<DatalogTerm, DatalogError> {
    let input = input.trim();

    if input.is_empty() {
        return Err(DatalogError::ParseError("empty term".to_string()));
    }

    // Quoted string constant
    if input.starts_with('"') && input.ends_with('"') && input.len() >= 2 {
        let content = &input[1..input.len() - 1];
        return Ok(DatalogTerm::Constant(DatalogValue::Str(
            content.to_string(),
        )));
    }

    // Check first character
    let first = input
        .chars()
        .next()
        .expect("non-empty input has a first char");

    if first.is_uppercase() || first == '_' {
        // Variable
        validate_identifier(input)?;
        return Ok(DatalogTerm::Variable(input.to_string()));
    }

    // Try boolean
    if input == "true" {
        return Ok(DatalogTerm::Constant(DatalogValue::Bool(true)));
    }
    if input == "false" {
        return Ok(DatalogTerm::Constant(DatalogValue::Bool(false)));
    }

    // Try integer
    if let Ok(i) = input.parse::<i64>() {
        return Ok(DatalogTerm::Constant(DatalogValue::Int(i)));
    }

    // Unquoted lowercase identifier → string constant
    if first.is_lowercase() || first == '_' {
        validate_identifier(input)?;
        return Ok(DatalogTerm::Constant(DatalogValue::Str(input.to_string())));
    }

    Err(DatalogError::ParseError(format!(
        "cannot parse term: {input}"
    )))
}

/// Validate that an identifier contains only alphanumeric chars and underscores.
fn validate_identifier(s: &str) -> Result<(), DatalogError> {
    for ch in s.chars() {
        if !ch.is_alphanumeric() && ch != '_' {
            return Err(DatalogError::ParseError(format!(
                "invalid identifier character '{ch}' in: {s}"
            )));
        }
    }
    Ok(())
}

/// Validate that a predicate name is a valid lowercase identifier.
fn validate_predicate(name: &str) -> Result<(), DatalogError> {
    if name.is_empty() {
        return Err(DatalogError::ParseError(
            "predicate name cannot be empty".to_string(),
        ));
    }
    let first = name
        .chars()
        .next()
        .expect("non-empty name has a first char");
    if first.is_uppercase() {
        return Err(DatalogError::ParseError(format!(
            "predicate name must start with lowercase: {name}"
        )));
    }
    validate_identifier(name)
}

#[cfg(test)]
mod parser_unit_tests {
    use super::*;

    #[test]
    fn test_strip_comments() {
        let input = "parent(a, b). % this is a comment\nparent(b, c).";
        let stripped = strip_comments(input);
        assert!(!stripped.contains('%'));
        assert!(stripped.contains("parent(a, b)"));
    }

    #[test]
    fn test_split_clauses_basic() {
        let input = "parent(a, b). parent(b, c).";
        let clauses = split_clauses(input);
        assert_eq!(clauses.len(), 2);
    }

    #[test]
    fn test_parse_term_variable() {
        let t = parse_term("X").expect("parse");
        assert_eq!(t, DatalogTerm::Variable("X".to_string()));
    }

    #[test]
    fn test_parse_term_string_constant() {
        let t = parse_term("alice").expect("parse");
        assert_eq!(
            t,
            DatalogTerm::Constant(DatalogValue::Str("alice".to_string()))
        );
    }

    #[test]
    fn test_parse_term_int() {
        let t = parse_term("42").expect("parse");
        assert_eq!(t, DatalogTerm::Constant(DatalogValue::Int(42)));
    }

    #[test]
    fn test_parse_term_bool() {
        let t = parse_term("true").expect("parse");
        assert_eq!(t, DatalogTerm::Constant(DatalogValue::Bool(true)));
    }
}
