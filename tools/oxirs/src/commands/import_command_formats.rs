//! # Import Command Formats
//!
//! Format-specific import handlers: N-Triples, N-Quads, Turtle, TriG, CSV, JSON-LD, RDF/XML.

use std::collections::HashMap;

use super::import_command_runner::{
    extract_graph_iri, extract_json_string_pair, extract_json_string_value, extract_xml_attr,
    find_matching_brace, find_obj_end, find_obj_start, parse_nt_term, parse_prefix_decl,
    parse_turtle_triple, tokenise_nt_line,
};
use super::import_command_types::{ImportError, ImportFormat, ImportResult, Triple};

// ---------------------------------------------------------------------------
// N-Triples
// ---------------------------------------------------------------------------

/// Parse N-Triples: `<subject> <predicate> <object> .` one per line.
pub(crate) fn parse_ntriples_impl(input: &str) -> Result<ImportResult, ImportError> {
    if input.trim().is_empty() {
        return Err(ImportError::EmptyInput);
    }
    let mut triples = Vec::new();
    let mut warnings = Vec::new();

    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_ntriples_line(line) {
            Ok(triple) => triples.push(triple),
            Err(msg) => warnings.push(format!("Line {}: {}", line_no + 1, msg)),
        }
    }

    Ok(ImportResult {
        triples,
        prefixes: HashMap::new(),
        graphs: Vec::new(),
        warnings,
        format_detected: ImportFormat::NTriples,
    })
}

fn parse_ntriples_line(line: &str) -> Result<Triple, String> {
    let tokens = tokenise_nt_line(line);
    if tokens.len() < 3 {
        return Err(format!(
            "expected 3 terms, got {} in: {}",
            tokens.len(),
            line
        ));
    }
    let subject = parse_nt_term(&tokens[0])?;
    let predicate = parse_nt_term(&tokens[1])?;
    let object = parse_nt_term(&tokens[2])?;
    Ok(Triple {
        subject,
        predicate,
        object,
        graph: None,
    })
}

// ---------------------------------------------------------------------------
// N-Quads
// ---------------------------------------------------------------------------

/// Parse N-Quads: `<s> <p> <o> [<g>] .` one per line.
pub(crate) fn parse_nquads_impl(input: &str) -> Result<ImportResult, ImportError> {
    if input.trim().is_empty() {
        return Err(ImportError::EmptyInput);
    }
    let mut triples = Vec::new();
    let mut warnings = Vec::new();
    let mut graphs: Vec<String> = Vec::new();

    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_nquads_line(line) {
            Ok(triple) => {
                if let Some(ref g) = triple.graph {
                    if !graphs.contains(g) {
                        graphs.push(g.clone());
                    }
                }
                triples.push(triple);
            }
            Err(msg) => warnings.push(format!("Line {}: {}", line_no + 1, msg)),
        }
    }

    Ok(ImportResult {
        triples,
        prefixes: HashMap::new(),
        graphs,
        warnings,
        format_detected: ImportFormat::NQuads,
    })
}

fn parse_nquads_line(line: &str) -> Result<Triple, String> {
    let tokens = tokenise_nt_line(line);
    if tokens.len() < 3 {
        return Err(format!("expected ≥ 3 terms, got {}", tokens.len()));
    }
    let subject = parse_nt_term(&tokens[0])?;
    let predicate = parse_nt_term(&tokens[1])?;
    let object = parse_nt_term(&tokens[2])?;
    let graph = if tokens.len() >= 4 && tokens[3] != "." {
        if tokens[3].starts_with('<') {
            Some(parse_nt_term(&tokens[3])?)
        } else {
            None
        }
    } else {
        None
    };
    Ok(Triple {
        subject,
        predicate,
        object,
        graph,
    })
}

// ---------------------------------------------------------------------------
// Turtle
// ---------------------------------------------------------------------------

/// Parse simplified Turtle: `@prefix` declarations followed by triples.
pub(crate) fn parse_turtle_impl(input: &str) -> Result<ImportResult, ImportError> {
    if input.trim().is_empty() {
        return Err(ImportError::EmptyInput);
    }
    let mut triples = Vec::new();
    let mut prefixes: HashMap<String, String> = HashMap::new();
    let mut warnings = Vec::new();

    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("@prefix") || line.starts_with("@base") {
            if let Some(ns) = parse_prefix_decl(line) {
                prefixes.insert(ns.0, ns.1);
            }
            continue;
        }
        match parse_turtle_triple(line, &prefixes) {
            Ok(Some(triple)) => triples.push(triple),
            Ok(None) => {}
            Err(msg) => warnings.push(format!("Line {}: {}", line_no + 1, msg)),
        }
    }

    Ok(ImportResult {
        triples,
        prefixes,
        graphs: Vec::new(),
        warnings,
        format_detected: ImportFormat::Turtle,
    })
}

// ---------------------------------------------------------------------------
// TriG
// ---------------------------------------------------------------------------

/// Parse simplified TriG: Turtle + `GRAPH <iri> { ... }` blocks.
pub(crate) fn parse_trig_impl(input: &str) -> Result<ImportResult, ImportError> {
    if input.trim().is_empty() {
        return Err(ImportError::EmptyInput);
    }
    let mut triples = Vec::new();
    let mut prefixes: HashMap<String, String> = HashMap::new();
    let mut graphs: Vec<String> = Vec::new();
    let mut warnings = Vec::new();
    let mut current_graph: Option<String> = None;

    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("@prefix") || line.starts_with("PREFIX") {
            if let Some(ns) = parse_prefix_decl(line) {
                prefixes.insert(ns.0, ns.1);
            }
            continue;
        }
        if line.to_uppercase().starts_with("GRAPH") {
            if let Some(graph_iri) = extract_graph_iri(line) {
                if !graphs.contains(&graph_iri) {
                    graphs.push(graph_iri.clone());
                }
                current_graph = Some(graph_iri);
            }
            continue;
        }
        if line == "}" {
            current_graph = None;
            continue;
        }
        match parse_turtle_triple(line, &prefixes) {
            Ok(Some(mut triple)) => {
                triple.graph = current_graph.clone();
                triples.push(triple);
            }
            Ok(None) => {}
            Err(msg) => warnings.push(format!("Line {}: {}", line_no + 1, msg)),
        }
    }

    Ok(ImportResult {
        triples,
        prefixes,
        graphs,
        warnings,
        format_detected: ImportFormat::TriG,
    })
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/// Parse CSV: first line is a header `subject,predicate,object[,graph]`.
pub(crate) fn parse_csv_impl(input: &str) -> Result<ImportResult, ImportError> {
    if input.trim().is_empty() {
        return Err(ImportError::EmptyInput);
    }
    let mut lines = input.lines();
    let header = lines.next().ok_or(ImportError::EmptyInput)?.trim();
    let cols: Vec<&str> = header.split(',').map(str::trim).collect();

    let has_graph = cols.len() >= 4 && cols[3].to_lowercase() == "graph";

    let mut triples = Vec::new();
    let mut warnings = Vec::new();
    let mut graphs: Vec<String> = Vec::new();

    for (row_no, raw_line) in lines.enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(4, ',').collect();
        if parts.len() < 3 {
            warnings.push(format!(
                "Row {}: expected at least 3 columns, got {}",
                row_no + 2,
                parts.len()
            ));
            continue;
        }
        let graph = if has_graph && parts.len() >= 4 {
            let g = parts[3].trim().to_string();
            if !g.is_empty() {
                if !graphs.contains(&g) {
                    graphs.push(g.clone());
                }
                Some(g)
            } else {
                None
            }
        } else {
            None
        };
        triples.push(Triple {
            subject: parts[0].trim().to_string(),
            predicate: parts[1].trim().to_string(),
            object: parts[2].trim().to_string(),
            graph,
        });
    }

    Ok(ImportResult {
        triples,
        prefixes: HashMap::new(),
        graphs,
        warnings,
        format_detected: ImportFormat::Csv,
    })
}

// ---------------------------------------------------------------------------
// JSON-LD
// ---------------------------------------------------------------------------

/// Parse simplified JSON-LD.
pub(crate) fn parse_jsonld_impl(input: &str) -> Result<ImportResult, ImportError> {
    if input.trim().is_empty() {
        return Err(ImportError::EmptyInput);
    }
    let mut triples = Vec::new();
    let mut prefixes: HashMap<String, String> = HashMap::new();
    let mut warnings = Vec::new();

    if let Some(ctx_start) = input.find("\"@context\"") {
        if let Some(brace_start) = input[ctx_start..].find('{') {
            let ctx_text = &input[ctx_start + brace_start..];
            if let Some(brace_end) = find_matching_brace(ctx_text) {
                let ctx_body = &ctx_text[1..brace_end];
                for line in ctx_body.lines() {
                    if let Some((k, v)) = extract_json_string_pair(line) {
                        if !k.starts_with('@') {
                            prefixes.insert(k, v);
                        }
                    }
                }
            }
        }
    }

    let graph_content = if let Some(pos) = input.find("\"@graph\"") {
        &input[pos..]
    } else {
        input
    };

    let mut search_pos = 0;
    while let Some(id_pos) = graph_content[search_pos..].find("\"@id\"") {
        let abs_id = search_pos + id_pos;
        let after_id = &graph_content[abs_id + 5..];
        let subject = match extract_json_string_value(after_id) {
            Some(v) => v,
            None => {
                search_pos = abs_id + 5;
                continue;
            }
        };

        let obj_start = find_obj_start(&graph_content[..abs_id]);
        let obj_end_rel = find_obj_end(&graph_content[abs_id..]).unwrap_or(100);
        let obj_end = abs_id + obj_end_rel;
        let obj_text = &graph_content[obj_start..obj_end.min(graph_content.len())];

        for line in obj_text.lines() {
            if let Some((key, value)) = extract_json_string_pair(line) {
                if key == "@id" || key.starts_with('@') {
                    continue;
                }
                let predicate = if key.contains(':') {
                    if key.starts_with("http") {
                        key.clone()
                    } else {
                        let colon = key.find(':').unwrap_or(key.len());
                        let pfx = &key[..colon];
                        let local = &key[colon + 1..];
                        if let Some(ns) = prefixes.get(pfx) {
                            format!("{}{}", ns, local)
                        } else {
                            key.clone()
                        }
                    }
                } else {
                    key.clone()
                };
                triples.push(Triple {
                    subject: subject.clone(),
                    predicate,
                    object: value,
                    graph: None,
                });
            }
        }

        search_pos = abs_id + 5;
    }

    if triples.is_empty() && !input.contains("@id") {
        warnings.push("No @id found — no triples extracted from JSON-LD".to_string());
    }

    Ok(ImportResult {
        triples,
        prefixes,
        graphs: Vec::new(),
        warnings,
        format_detected: ImportFormat::JsonLd,
    })
}

// ---------------------------------------------------------------------------
// RDF/XML
// ---------------------------------------------------------------------------

/// Parse simplified RDF/XML: looks for `rdf:Description` elements.
pub(crate) fn parse_rdfxml_impl(input: &str) -> Result<ImportResult, ImportError> {
    if input.trim().is_empty() {
        return Err(ImportError::EmptyInput);
    }
    let mut triples = Vec::new();
    let mut warnings = Vec::new();

    let mut pos = 0;
    while let Some(desc_pos) = input[pos..].find("rdf:Description") {
        let abs = pos + desc_pos;
        let tag_end = input[abs..]
            .find('>')
            .map(|p| abs + p)
            .unwrap_or(input.len());
        let tag_text = &input[abs..tag_end];
        let subject = extract_xml_attr(tag_text, "rdf:about")
            .or_else(|| extract_xml_attr(tag_text, "about"))
            .unwrap_or_else(|| "_:blank".to_string());

        let close_tag = "</rdf:Description>";
        let block_end = input[abs..]
            .find(close_tag)
            .map(|p| abs + p)
            .unwrap_or(input.len());
        let block_text = &input[abs..block_end];

        let mut child_pos = 0;
        while let Some(elem_start) = block_text[child_pos..].find('<') {
            let abs_elem = child_pos + elem_start;
            if block_text[abs_elem..].starts_with("<rdf:Description") {
                break;
            }
            if block_text[abs_elem..].starts_with("</") {
                child_pos = abs_elem + 2;
                continue;
            }
            let rest = &block_text[abs_elem + 1..];
            let name_end = rest
                .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
                .unwrap_or(rest.len());
            let tag_name = &rest[..name_end];
            if tag_name.is_empty() {
                child_pos = abs_elem + 1;
                continue;
            }
            let open_end = block_text[abs_elem..]
                .find('>')
                .map(|p| abs_elem + p + 1)
                .unwrap_or(block_text.len());
            let close_pat = format!("</{}>", tag_name);
            let content_end = block_text[open_end..]
                .find(&close_pat)
                .map(|p| open_end + p);
            if let Some(end) = content_end {
                let content = block_text[open_end..end].trim();
                if !content.is_empty() && !tag_name.starts_with("rdf:") {
                    triples.push(Triple {
                        subject: subject.clone(),
                        predicate: tag_name.to_string(),
                        object: content.to_string(),
                        graph: None,
                    });
                }
                child_pos = end;
            } else {
                child_pos = open_end;
            }
        }

        pos = block_end + close_tag.len();
        if pos >= input.len() {
            break;
        }
    }

    if triples.is_empty() {
        warnings.push("No rdf:Description elements found".to_string());
    }

    Ok(ImportResult {
        triples,
        prefixes: HashMap::new(),
        graphs: Vec::new(),
        warnings,
        format_detected: ImportFormat::RdfXml,
    })
}
