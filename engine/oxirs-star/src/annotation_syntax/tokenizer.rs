//! Tokenizer for RDF-star annotation blocks `{| ... |}`
use super::AnnotationSyntaxError;

/// Tokens produced by the annotation block tokenizer
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationToken {
    /// `{|` — open annotation block
    LBrace2,
    /// `|}` — close annotation block
    RBrace2,
    /// An IRI: the string inside `< ... >`
    NamedNode(String),
    /// A blank node identifier: the label after `_:`
    BlankNode(String),
    /// A string literal: (value, optional language tag, optional datatype URI)
    Literal(String, Option<String>, Option<String>),
    /// `.` separator
    Dot,
    /// `;` separator
    Semicolon,
    /// `,` separator
    Comma,
}

/// Tokenize the body of a `{| ... |}` annotation block.
///
/// The input should be the text between `{|` and `|}` (not including those
/// delimiters themselves).
pub fn tokenize_annotation_block(
    input: &str,
) -> Result<Vec<AnnotationToken>, AnnotationSyntaxError> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        match c {
            // Skip whitespace
            c if c.is_whitespace() => continue,

            // Comment until end of line (# ...)
            '#' => {
                for (_, ch) in chars.by_ref() {
                    if ch == '\n' {
                        break;
                    }
                }
            }

            // Open annotation block `{|`
            '{' => {
                if chars.peek().map(|(_, ch)| *ch) == Some('|') {
                    chars.next(); // consume '|'
                    tokens.push(AnnotationToken::LBrace2);
                } else {
                    return Err(AnnotationSyntaxError::UnexpectedToken {
                        expected: "{| (annotation block open)".to_string(),
                        got: format!("'{{' at position {}", i),
                    });
                }
            }

            // Close annotation block `|}`
            '|' => {
                if chars.peek().map(|(_, ch)| *ch) == Some('}') {
                    chars.next(); // consume '}'
                    tokens.push(AnnotationToken::RBrace2);
                } else {
                    return Err(AnnotationSyntaxError::UnexpectedToken {
                        expected: "|} (annotation block close)".to_string(),
                        got: format!("'|' at position {}", i),
                    });
                }
            }

            // Named node `<...>`
            '<' => {
                let iri = read_iri(&mut chars, i)?;
                tokens.push(AnnotationToken::NamedNode(iri));
            }

            // Blank node `_:label`
            '_' => {
                if chars.peek().map(|(_, ch)| *ch) == Some(':') {
                    chars.next(); // consume ':'
                    let label = read_label(&mut chars);
                    tokens.push(AnnotationToken::BlankNode(label));
                } else {
                    return Err(AnnotationSyntaxError::UnexpectedToken {
                        expected: "_: (blank node prefix)".to_string(),
                        got: format!("'_' without ':' at position {}", i),
                    });
                }
            }

            // String literal `"..."`
            '"' => {
                let (value, lang, datatype) = read_literal(&mut chars, i)?;
                tokens.push(AnnotationToken::Literal(value, lang, datatype));
            }

            '.' => tokens.push(AnnotationToken::Dot),
            ';' => tokens.push(AnnotationToken::Semicolon),
            ',' => tokens.push(AnnotationToken::Comma),

            other => {
                return Err(AnnotationSyntaxError::UnexpectedToken {
                    expected: "IRI, blank node, literal, or separator".to_string(),
                    got: format!("'{}' at position {}", other, i),
                });
            }
        }
    }

    Ok(tokens)
}

/// Find byte positions of all `{| ... |}` annotation blocks in Turtle text.
///
/// Returns a `Vec<(start, end)>` where `start` is the byte index of `{|`
/// and `end` is the byte index just after `|}`.
pub fn find_annotation_blocks(input: &str) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i + 1 < len {
        // Look for `{|`
        if bytes[i] == b'{' && bytes[i + 1] == b'|' {
            let start = i;
            i += 2; // skip `{|`

            // Scan forward for matching `|}`; handle nested annotation blocks
            let mut depth = 1usize;
            while i + 1 < len {
                if bytes[i] == b'{' && bytes[i + 1] == b'|' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'|' && bytes[i + 1] == b'}' {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        result.push((start, i));
                        break;
                    }
                } else if bytes[i] == b'"' {
                    // Skip over string literals to avoid false positives
                    i += 1;
                    while i < len {
                        if bytes[i] == b'\\' {
                            i += 2; // skip escape sequence
                        } else if bytes[i] == b'"' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                } else if bytes[i] == b'#' {
                    // Skip line comment
                    while i < len && bytes[i] != b'\n' {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
    }

    result
}

// ---- private helpers ----

fn read_iri(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    start: usize,
) -> Result<String, AnnotationSyntaxError> {
    let mut iri = String::new();
    for (_, c) in chars.by_ref() {
        if c == '>' {
            return Ok(iri);
        }
        iri.push(c);
    }
    Err(AnnotationSyntaxError::UnexpectedToken {
        expected: "closing '>' for IRI".to_string(),
        got: format!("end of input after position {}", start),
    })
}

fn read_label(chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>) -> String {
    let mut label = String::new();
    while let Some((_, c)) = chars.peek() {
        if c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.' {
            label.push(*c);
            chars.next();
        } else {
            break;
        }
    }
    label
}

fn read_literal(
    chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>,
    start: usize,
) -> Result<(String, Option<String>, Option<String>), AnnotationSyntaxError> {
    let mut value = String::new();
    let mut escaped = false;

    // Read until closing '"'
    loop {
        match chars.next() {
            None => {
                return Err(AnnotationSyntaxError::UnexpectedToken {
                    expected: "closing '\"' for string literal".to_string(),
                    got: format!("end of input after position {}", start),
                });
            }
            Some((_, c)) => {
                if escaped {
                    value.push(match c {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '"' => '"',
                        '\\' => '\\',
                        other => other,
                    });
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    break;
                } else {
                    value.push(c);
                }
            }
        }
    }

    // Check for language tag `@lang` or datatype `^^<uri>`
    match chars.peek() {
        Some((_, '@')) => {
            chars.next(); // consume '@'
            let lang = read_label(chars);
            Ok((value, Some(lang), None))
        }
        Some((_, '^')) => {
            chars.next(); // consume first '^'
            if chars.peek().map(|(_, c)| *c) == Some('^') {
                chars.next(); // consume second '^'
            }
            // Expect `<datatype-iri>`
            match chars.next() {
                Some((pos, '<')) => {
                    let dt = read_iri(chars, pos)?;
                    Ok((value, None, Some(dt)))
                }
                Some((_, other)) => Err(AnnotationSyntaxError::UnexpectedToken {
                    expected: "<datatype IRI> after ^^".to_string(),
                    got: format!("'{}'", other),
                }),
                None => Err(AnnotationSyntaxError::UnexpectedToken {
                    expected: "<datatype IRI> after ^^".to_string(),
                    got: "end of input".to_string(),
                }),
            }
        }
        _ => Ok((value, None, None)),
    }
}
