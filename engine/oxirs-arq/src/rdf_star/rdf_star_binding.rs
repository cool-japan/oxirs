//! SPARQL-star pattern binding, CONSTRUCT instantiation, and built-in functions.
//!
//! Provides [`StarBinding`] and [`bind_pattern`] for matching quoted-triple
//! patterns against concrete triples, [`instantiate_quoted_triple`] for
//! substituting bindings into a CONSTRUCT template, and the
//! [`sparql_star_builtins`] module with `TRIPLE()`, `isTRIPLE()`, `SUBJECT()`,
//! `PREDICATE()`, and `OBJECT()`.

use crate::rdf_star::rdf_star_terms::{QuotedTriple, StarObject, StarPredicate, StarSubject};
use anyhow::anyhow;
use std::collections::HashMap;

// ─── Binding result ───────────────────────────────────────────────────────────

/// A variable binding produced by a SPARQL-star pattern match
pub type StarBinding = HashMap<String, StarObject>;

/// Bind variables in `pattern` to values in `triple` and push the result into `out`.
/// Returns `false` if the pattern does not match `triple`.
pub fn bind_pattern(triple: &QuotedTriple, pattern: &QuotedTriple, out: &mut StarBinding) -> bool {
    bind_subject(pattern, triple, out)
        && bind_predicate(pattern, triple, out)
        && bind_object(pattern, triple, out)
}

fn bind_subject(pattern: &QuotedTriple, triple: &QuotedTriple, out: &mut StarBinding) -> bool {
    match &pattern.subject {
        StarSubject::Variable(v) => {
            let obj = star_subject_to_object(&triple.subject);
            out.insert(v.as_str().to_string(), obj);
            true
        }
        StarSubject::NamedNode(pn) => {
            matches!(&triple.subject, StarSubject::NamedNode(vn) if vn == pn)
        }
        StarSubject::BlankNode(pb) => {
            matches!(&triple.subject, StarSubject::BlankNode(vb) if vb == pb)
        }
        StarSubject::Quoted(pq) => {
            if let StarSubject::Quoted(vq) = &triple.subject {
                bind_pattern(vq, pq, out)
            } else {
                false
            }
        }
    }
}

fn bind_predicate(pattern: &QuotedTriple, triple: &QuotedTriple, out: &mut StarBinding) -> bool {
    match &pattern.predicate {
        StarPredicate::Variable(v) => {
            if let StarPredicate::NamedNode(n) = &triple.predicate {
                out.insert(v.as_str().to_string(), StarObject::NamedNode(n.clone()));
            }
            true
        }
        StarPredicate::NamedNode(pn) => {
            matches!(&triple.predicate, StarPredicate::NamedNode(vn) if vn == pn)
        }
    }
}

fn bind_object(pattern: &QuotedTriple, triple: &QuotedTriple, out: &mut StarBinding) -> bool {
    match &pattern.object {
        StarObject::Variable(v) => {
            out.insert(v.as_str().to_string(), triple.object.clone());
            true
        }
        StarObject::NamedNode(pn) => {
            matches!(&triple.object, StarObject::NamedNode(vn) if vn == pn)
        }
        StarObject::BlankNode(pb) => {
            matches!(&triple.object, StarObject::BlankNode(vb) if vb == pb)
        }
        StarObject::Literal(pl) => {
            matches!(&triple.object, StarObject::Literal(vl) if vl == pl)
        }
        StarObject::Quoted(pq) => {
            if let StarObject::Quoted(vq) = &triple.object {
                bind_pattern(vq, pq, out)
            } else {
                false
            }
        }
    }
}

/// Convert a [`StarSubject`] to a [`StarObject`] for binding
fn star_subject_to_object(subject: &StarSubject) -> StarObject {
    match subject {
        StarSubject::NamedNode(n) => StarObject::NamedNode(n.clone()),
        StarSubject::BlankNode(id) => StarObject::BlankNode(id.clone()),
        StarSubject::Variable(v) => StarObject::Variable(v.clone()),
        StarSubject::Quoted(qt) => StarObject::Quoted(qt.clone()),
    }
}

// ─── CONSTRUCT helpers ────────────────────────────────────────────────────────

/// Apply a [`StarBinding`] to a [`QuotedTriple`] template, substituting variables
pub fn instantiate_quoted_triple(
    template: &QuotedTriple,
    binding: &StarBinding,
) -> anyhow::Result<QuotedTriple> {
    let subject = instantiate_subject(&template.subject, binding)?;
    let predicate = instantiate_predicate(&template.predicate, binding)?;
    let object = instantiate_object(&template.object, binding)?;
    Ok(QuotedTriple::new(subject, predicate, object))
}

fn instantiate_subject(s: &StarSubject, binding: &StarBinding) -> anyhow::Result<StarSubject> {
    match s {
        StarSubject::Variable(v) => {
            let val = binding
                .get(v.as_str())
                .ok_or_else(|| anyhow!("unbound variable ?{}", v.as_str()))?;
            // Attempt to convert the bound StarObject back to StarSubject
            object_to_subject(val)
        }
        StarSubject::Quoted(qt) => Ok(StarSubject::Quoted(Box::new(instantiate_quoted_triple(
            qt, binding,
        )?))),
        other => Ok(other.clone()),
    }
}

fn instantiate_predicate(
    p: &StarPredicate,
    binding: &StarBinding,
) -> anyhow::Result<StarPredicate> {
    match p {
        StarPredicate::Variable(v) => {
            let val = binding
                .get(v.as_str())
                .ok_or_else(|| anyhow!("unbound predicate variable ?{}", v.as_str()))?;
            match val {
                StarObject::NamedNode(n) => Ok(StarPredicate::NamedNode(n.clone())),
                other => Err(anyhow!("predicate must be a named node, got {other}")),
            }
        }
        other => Ok(other.clone()),
    }
}

fn instantiate_object(o: &StarObject, binding: &StarBinding) -> anyhow::Result<StarObject> {
    match o {
        StarObject::Variable(v) => binding
            .get(v.as_str())
            .cloned()
            .ok_or_else(|| anyhow!("unbound variable ?{}", v.as_str())),
        StarObject::Quoted(qt) => Ok(StarObject::Quoted(Box::new(instantiate_quoted_triple(
            qt, binding,
        )?))),
        other => Ok(other.clone()),
    }
}

fn object_to_subject(obj: &StarObject) -> anyhow::Result<StarSubject> {
    match obj {
        StarObject::NamedNode(n) => Ok(StarSubject::NamedNode(n.clone())),
        StarObject::BlankNode(id) => Ok(StarSubject::BlankNode(id.clone())),
        StarObject::Quoted(qt) => Ok(StarSubject::Quoted(qt.clone())),
        StarObject::Literal(_) => Err(anyhow!("literals cannot be subjects")),
        StarObject::Variable(v) => Ok(StarSubject::Variable(v.clone())),
    }
}

// ─── SPARQL-star functions ────────────────────────────────────────────────────

/// SPARQL-star built-in functions: `TRIPLE()`, `isTRIPLE()`, `SUBJECT()`,
/// `PREDICATE()`, `OBJECT()`
pub mod sparql_star_builtins {
    use crate::rdf_star::rdf_star_terms::{QuotedTriple, StarObject, StarPredicate, StarSubject};

    /// `TRIPLE(s, p, o)` — construct a quoted triple from three terms
    pub fn triple_fn(
        subject: StarSubject,
        predicate: StarPredicate,
        object: StarObject,
    ) -> QuotedTriple {
        QuotedTriple::new(subject, predicate, object)
    }

    /// `isTRIPLE(term)` — return `true` if the term is a quoted triple
    pub fn is_triple(obj: &StarObject) -> bool {
        matches!(obj, StarObject::Quoted(_))
    }

    /// `SUBJECT(triple)` — extract the subject of a quoted triple
    pub fn subject_of(qt: &QuotedTriple) -> &StarSubject {
        &qt.subject
    }

    /// `PREDICATE(triple)` — extract the predicate of a quoted triple
    pub fn predicate_of(qt: &QuotedTriple) -> &StarPredicate {
        &qt.predicate
    }

    /// `OBJECT(triple)` — extract the object of a quoted triple
    pub fn object_of(qt: &QuotedTriple) -> &StarObject {
        &qt.object
    }
}
