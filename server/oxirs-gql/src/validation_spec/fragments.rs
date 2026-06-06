//! Fragment validation rules.
//!
//! Covers: UniqueFragmentNames, KnownFragmentNames, NoFragmentCycles, NoUnusedFragments.

use std::collections::{HashMap, HashSet};

use crate::ast::{Definition, Document, FragmentDefinition, Selection, SelectionSet};
use crate::validation::{SpecRule, ValidationError, ValidationRule};

use super::operations::{collect_fragments, collect_spread_names};

// ─────────────────────────────────────────────────────────────────────────────
// UniqueFragmentNames
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_unique_fragment_names(doc: &Document) -> Vec<ValidationError> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut errors = Vec::new();

    for def in &doc.definitions {
        if let Definition::Fragment(frag) = def {
            let count = seen.entry(frag.name.clone()).or_insert(0);
            *count += 1;
            if *count == 2 {
                errors.push(ValidationError::new(
                    format!("There can be only one fragment named \"{}\".", frag.name),
                    ValidationRule::Spec(SpecRule::UniqueFragmentNames),
                ));
            }
        }
    }

    errors
}

// ─────────────────────────────────────────────────────────────────────────────
// KnownFragmentNames
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_known_fragment_names(doc: &Document) -> Vec<ValidationError> {
    let defined: HashSet<String> = doc
        .definitions
        .iter()
        .filter_map(|d| {
            if let Definition::Fragment(f) = d {
                Some(f.name.clone())
            } else {
                None
            }
        })
        .collect();

    let mut errors = Vec::new();
    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            collect_unknown_fragment_errors(&op.selection_set, &defined, &mut errors);
        }
    }

    errors
}

fn collect_unknown_fragment_errors(
    ss: &SelectionSet,
    defined: &HashSet<String>,
    errors: &mut Vec<ValidationError>,
) {
    for sel in &ss.selections {
        match sel {
            Selection::FragmentSpread(spread) => {
                if !defined.contains(&spread.fragment_name) {
                    errors.push(ValidationError::new(
                        format!(
                            "Unknown fragment \"{}\". Did you mean to define it?",
                            spread.fragment_name
                        ),
                        ValidationRule::Spec(SpecRule::KnownFragmentNames),
                    ));
                }
            }
            Selection::InlineFragment(inl) => {
                collect_unknown_fragment_errors(&inl.selection_set, defined, errors);
            }
            Selection::Field(f) => {
                if let Some(ss2) = &f.selection_set {
                    collect_unknown_fragment_errors(ss2, defined, errors);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NoFragmentCycles
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_no_fragment_cycles(doc: &Document) -> Vec<ValidationError> {
    let fragments = collect_fragments(doc);

    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for (name, frag) in &fragments {
        let mut deps = Vec::new();
        collect_spread_names(&frag.selection_set, &mut deps);
        adjacency.insert(name.to_string(), deps);
    }

    let mut errors = Vec::new();
    let mut color: HashMap<String, u8> = HashMap::new();

    for name in adjacency.keys().cloned().collect::<Vec<_>>() {
        if !color.contains_key(&name) {
            dfs_cycle_detect(&name, &adjacency, &mut color, &mut errors, &mut Vec::new());
        }
    }

    errors
}

fn dfs_cycle_detect(
    node: &str,
    adj: &HashMap<String, Vec<String>>,
    color: &mut HashMap<String, u8>,
    errors: &mut Vec<ValidationError>,
    stack: &mut Vec<String>,
) {
    color.insert(node.to_string(), 1);
    stack.push(node.to_string());

    if let Some(neighbors) = adj.get(node) {
        for neighbor in neighbors {
            match color.get(neighbor.as_str()).copied().unwrap_or(0) {
                1 => {
                    let cycle_start = stack.iter().position(|s| s == neighbor).unwrap_or(0);
                    let cycle_path: Vec<&str> =
                        stack[cycle_start..].iter().map(|s| s.as_str()).collect();
                    errors.push(ValidationError::new(
                        format!(
                            "Cannot spread fragment \"{}\" within itself via {}.",
                            neighbor,
                            cycle_path.join(" \u{2192} ")
                        ),
                        ValidationRule::Spec(SpecRule::NoFragmentCycles),
                    ));
                }
                0 => {
                    dfs_cycle_detect(neighbor, adj, color, errors, stack);
                }
                _ => {}
            }
        }
    }

    stack.pop();
    color.insert(node.to_string(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// NoUnusedFragments
// ─────────────────────────────────────────────────────────────────────────────

pub(super) fn check_no_unused_fragments(doc: &Document) -> Vec<ValidationError> {
    let defined: HashMap<String, &FragmentDefinition> = collect_fragments(doc);
    let mut used: HashSet<String> = HashSet::new();

    for def in &doc.definitions {
        if let Definition::Operation(op) = def {
            collect_used_fragments(&op.selection_set, &mut used, &defined);
        }
    }

    defined
        .keys()
        .filter(|name| !used.contains(*name))
        .map(|name| {
            ValidationError::new(
                format!("Fragment \"{}\" is never used.", name),
                ValidationRule::Spec(SpecRule::NoUnusedFragments),
            )
        })
        .collect()
}

fn collect_used_fragments(
    ss: &SelectionSet,
    used: &mut HashSet<String>,
    defined: &HashMap<String, &FragmentDefinition>,
) {
    for sel in &ss.selections {
        match sel {
            Selection::FragmentSpread(spread) => {
                if !used.contains(&spread.fragment_name) {
                    used.insert(spread.fragment_name.clone());
                    if let Some(frag) = defined.get(&spread.fragment_name) {
                        collect_used_fragments(&frag.selection_set, used, defined);
                    }
                }
            }
            Selection::InlineFragment(inl) => {
                collect_used_fragments(&inl.selection_set, used, defined);
            }
            Selection::Field(f) => {
                if let Some(ss2) = &f.selection_set {
                    collect_used_fragments(ss2, used, defined);
                }
            }
        }
    }
}
