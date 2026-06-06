//! Expand annotation shorthand to explicit RDF-star triples and serialize
use super::{AnnotatedTriple, AnnotationValue, RdfStarTriple, StarTerm};

/// Expand an `AnnotatedTriple` into explicit RDF-star triples.
///
/// Returns one entry per annotation pair, each being:
/// `(QuotedTriple(base), annotation_predicate, annotation_object)`.
///
/// That is, for:
/// ```text
/// <s> <p> <o> {| <ap1> <ao1> ; <ap2> <ao2> |}
/// ```
/// This returns:
/// ```text
/// [
///   (StarTerm::QuotedTriple(<< <s> <p> <o> >>), "<ap1>", AnnotationValue::NamedNode("<ao1>")),
///   (StarTerm::QuotedTriple(<< <s> <p> <o> >>), "<ap2>", AnnotationValue::NamedNode("<ao2>")),
/// ]
/// ```
pub fn expand_annotations(annotated: &AnnotatedTriple) -> Vec<(StarTerm, String, AnnotationValue)> {
    let quoted = StarTerm::QuotedTriple(Box::new(annotated.base.clone()));
    annotated
        .annotations
        .iter()
        .map(|pair| (quoted.clone(), pair.predicate.clone(), pair.object.clone()))
        .collect()
}

/// Serialize an `AnnotatedTriple` back to canonical Turtle-star using the
/// `{| ... |}` shorthand form.
pub fn annotations_to_turtle(annotated: &AnnotatedTriple) -> String {
    let base = format!(
        "{} {} {}",
        annotated.base.subject.to_turtle(),
        annotated.base.predicate.to_turtle(),
        annotated.base.object.to_turtle()
    );

    if annotated.annotations.is_empty() {
        return format!("{} .", base);
    }

    let annotation_body = annotated
        .annotations
        .iter()
        .map(|pair| format!("<{}> {}", pair.predicate, pair.object.to_turtle()))
        .collect::<Vec<_>>()
        .join(" ; ");

    format!("{} {{| {} |}} .", base, annotation_body)
}

/// Serialize an `AnnotatedTriple` to explicit `<< ... >> pred obj .` form
/// (no `{| ... |}` shorthand).
pub fn to_explicit_turtle(annotated: &AnnotatedTriple) -> String {
    let quoted = annotated.base.to_quoted();
    let mut lines = Vec::new();

    // First output the base triple
    lines.push(annotated.base.to_turtle());

    // Then output each annotation as an explicit quoted-triple statement
    for pair in &annotated.annotations {
        lines.push(format!(
            "{} <{}> {} .",
            quoted,
            pair.predicate,
            pair.object.to_turtle()
        ));
    }

    lines.join("\n")
}

/// Convenience: convert an `AnnotatedTriple` whose base is a quoted triple.
///
/// Given a quoted-triple subject of the form `<< s p o >>`, reconstructs a
/// synthetic `RdfStarTriple` for serialization.
pub fn quoted_triple_to_turtle(triple: &RdfStarTriple) -> String {
    triple.to_quoted()
}
