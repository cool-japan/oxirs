//! Type checking, conversion, and boolean functions for SPARQL 1.2

use crate::model::{BlankNode, Literal, NamedNode, Term};
use crate::OxirsError;

// Type conversion functions

/// STR - Convert to string
pub(super) fn fn_str(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "STR requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => Ok(Term::Literal(Literal::new(lit.value()))),
        Term::NamedNode(nn) => Ok(Term::Literal(Literal::new(nn.as_str()))),
        _ => Err(OxirsError::Query("STR requires literal or IRI".to_string())),
    }
}

/// LANG - Get language tag
pub(super) fn fn_lang(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "LANG requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let lang = lit.language().unwrap_or("");
            Ok(Term::Literal(Literal::new(lang)))
        }
        _ => Err(OxirsError::Query("LANG requires literal".to_string())),
    }
}

/// DATATYPE - Get datatype IRI
pub(super) fn fn_datatype(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "DATATYPE requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let dt = lit.datatype();
            Ok(Term::NamedNode(NamedNode::new(dt.as_str()).unwrap()))
        }
        _ => Err(OxirsError::Query("DATATYPE requires literal".to_string())),
    }
}

/// IRI - Convert to IRI
pub(super) fn fn_iri(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "IRI requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let iri = NamedNode::new(lit.value())?;
            Ok(Term::NamedNode(iri))
        }
        Term::NamedNode(nn) => Ok(Term::NamedNode(nn.clone())),
        _ => Err(OxirsError::Query(
            "IRI requires string literal or IRI".to_string(),
        )),
    }
}

/// BNODE - Create blank node
pub(super) fn fn_bnode(args: &[Term]) -> Result<Term, OxirsError> {
    if args.is_empty() {
        Ok(Term::BlankNode(BlankNode::new_unique()))
    } else if args.len() == 1 {
        match &args[0] {
            Term::Literal(lit) => {
                let bnode = BlankNode::new(lit.value())?;
                Ok(Term::BlankNode(bnode))
            }
            _ => Err(OxirsError::Query(
                "BNODE requires string literal or no arguments".to_string(),
            )),
        }
    } else {
        Err(OxirsError::Query(
            "BNODE requires 0 or 1 arguments".to_string(),
        ))
    }
}

/// STRDT - Create typed literal
pub(super) fn fn_strdt(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRDT requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(value_lit), Term::NamedNode(datatype)) => Ok(Term::Literal(
            Literal::new_typed(value_lit.value(), datatype.clone()),
        )),
        _ => Err(OxirsError::Query(
            "STRDT requires string literal and IRI".to_string(),
        )),
    }
}

/// STRLANG - Create language-tagged literal
pub(super) fn fn_strlang(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "STRLANG requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(value_lit), Term::Literal(lang_lit)) => Ok(Term::Literal(
            Literal::new_lang(value_lit.value(), lang_lit.value())?,
        )),
        _ => Err(OxirsError::Query(
            "STRLANG requires two string literals".to_string(),
        )),
    }
}

/// UUID - Generate UUID IRI
pub(super) fn fn_uuid(_args: &[Term]) -> Result<Term, OxirsError> {
    use uuid::Uuid;
    let uuid = Uuid::new_v4();
    let iri = NamedNode::new(format!("urn:uuid:{uuid}"))?;
    Ok(Term::NamedNode(iri))
}

/// STRUUID - Generate UUID string
pub(super) fn fn_struuid(_args: &[Term]) -> Result<Term, OxirsError> {
    use uuid::Uuid;
    let uuid = Uuid::new_v4();
    Ok(Term::Literal(Literal::new(uuid.to_string())))
}

// Boolean functions

/// NOT - Logical NOT
pub(super) fn fn_not(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "NOT requires exactly 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(lit) => {
            let value = lit.value() == "true";
            Ok(Term::Literal(Literal::new_typed(
                if !value { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "NOT requires boolean literal".to_string(),
        )),
    }
}

/// EXISTS - Check pattern existence
pub(super) fn fn_exists(_args: &[Term]) -> Result<Term, OxirsError> {
    // EXISTS needs special handling in query evaluation
    Err(OxirsError::Query(
        "EXISTS requires graph pattern context".to_string(),
    ))
}

/// NOT_EXISTS - Check pattern non-existence
pub(super) fn fn_not_exists(_args: &[Term]) -> Result<Term, OxirsError> {
    // NOT EXISTS needs special handling in query evaluation
    Err(OxirsError::Query(
        "NOT EXISTS requires graph pattern context".to_string(),
    ))
}

/// BOUND - Check if variable is bound
pub(super) fn fn_bound(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "BOUND requires exactly 1 argument".to_string(),
        ));
    }

    let is_bound = !matches!(&args[0], Term::Variable(_));
    Ok(Term::Literal(Literal::new_typed(
        if is_bound { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

/// COALESCE - Return first bound argument
pub(super) fn fn_coalesce(args: &[Term]) -> Result<Term, OxirsError> {
    for arg in args {
        if !matches!(arg, Term::Variable(_)) {
            return Ok(arg.clone());
        }
    }
    Err(OxirsError::Query(
        "COALESCE: all arguments are unbound".to_string(),
    ))
}

/// IF - Conditional expression
pub(super) fn fn_if(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 3 {
        return Err(OxirsError::Query(
            "IF requires exactly 3 arguments".to_string(),
        ));
    }

    match &args[0] {
        Term::Literal(condition) => {
            let is_true = condition.value() == "true";
            Ok(if is_true {
                args[1].clone()
            } else {
                args[2].clone()
            })
        }
        _ => Err(OxirsError::Query(
            "IF condition must be boolean".to_string(),
        )),
    }
}

// Type checking functions

/// isIRI - Check if term is IRI
#[allow(dead_code)]
pub(super) fn fn_is_iri(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "isIRI requires exactly 1 argument".to_string(),
        ));
    }

    let result = matches!(&args[0], Term::NamedNode(_));
    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

/// isBLANK - Check if term is blank node
#[allow(dead_code)]
pub(super) fn fn_is_blank(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "isBLANK requires exactly 1 argument".to_string(),
        ));
    }

    let result = matches!(&args[0], Term::BlankNode(_));
    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

/// isLITERAL - Check if term is literal
#[allow(dead_code)]
pub(super) fn fn_is_literal(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "isLITERAL requires exactly 1 argument".to_string(),
        ));
    }

    let result = matches!(&args[0], Term::Literal(_));
    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

/// isNUMERIC - Check if term is numeric
#[allow(dead_code)]
pub(super) fn fn_is_numeric(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 1 {
        return Err(OxirsError::Query(
            "isNUMERIC requires exactly 1 argument".to_string(),
        ));
    }

    let result = match &args[0] {
        Term::Literal(lit) => {
            let dt = lit.datatype();
            dt.as_str() == "http://www.w3.org/2001/XMLSchema#integer"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#decimal"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#float"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#double"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#nonPositiveInteger"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#negativeInteger"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#long"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#int"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#short"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#byte"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#nonNegativeInteger"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#unsignedLong"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#unsignedInt"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#unsignedShort"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#unsignedByte"
                || dt.as_str() == "http://www.w3.org/2001/XMLSchema#positiveInteger"
        }
        _ => false,
    };

    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

/// sameTerm - Check if two terms are the same
#[allow(dead_code)]
pub(super) fn fn_same_term(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "sameTerm requires exactly 2 arguments".to_string(),
        ));
    }

    let result = args[0] == args[1];
    Ok(Term::Literal(Literal::new_typed(
        if result { "true" } else { "false" },
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

/// LANGMATCHES - Match language tag
#[allow(dead_code)]
pub(super) fn fn_langmatches(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() != 2 {
        return Err(OxirsError::Query(
            "LANGMATCHES requires exactly 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Term::Literal(lang_tag), Term::Literal(lang_range)) => {
            let tag = lang_tag.value().to_lowercase();
            let range = lang_range.value().to_lowercase();

            let result = if range == "*" {
                !tag.is_empty()
            } else {
                tag == range || tag.starts_with(&format!("{}-", range))
            };

            Ok(Term::Literal(Literal::new_typed(
                if result { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => Err(OxirsError::Query(
            "LANGMATCHES requires two string literals".to_string(),
        )),
    }
}

// List functions

/// IN - Check if value is in list
pub(super) fn fn_in(args: &[Term]) -> Result<Term, OxirsError> {
    if args.len() < 2 {
        return Err(OxirsError::Query(
            "IN requires at least 2 arguments".to_string(),
        ));
    }

    let value = &args[0];
    for item in &args[1..] {
        if value == item {
            return Ok(Term::Literal(Literal::new_typed(
                "true",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )));
        }
    }

    Ok(Term::Literal(Literal::new_typed(
        "false",
        NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
    )))
}

/// NOT_IN - Check if value is not in list
pub(super) fn fn_not_in(args: &[Term]) -> Result<Term, OxirsError> {
    match fn_in(args)? {
        Term::Literal(lit) => {
            let value = lit.value() == "true";
            Ok(Term::Literal(Literal::new_typed(
                if !value { "true" } else { "false" },
                NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").unwrap(),
            )))
        }
        _ => unreachable!(),
    }
}
