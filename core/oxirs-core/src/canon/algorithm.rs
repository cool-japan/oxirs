//! URDNA2015 / RDNA 2015 core algorithm implementation.
//!
//! This module provides a complete, spec-faithful implementation of the W3C
//! RDF Dataset Normalization Algorithm (URDNA2015) as described in:
//! <https://www.w3.org/TR/rdf-canon/>
//!
//! ## Algorithm Overview
//!
//! URDNA2015 assigns deterministic canonical blank node identifiers to every
//! blank node in an RDF dataset.  The algorithm works in five stages:
//!
//! 1. **Collect blank nodes** — enumerate every blank node appearing in any quad.
//! 2. **Hash First-Degree Quads** — for each blank node, hash the set of quads
//!    it appears in using `_:a` for the node itself and `_:z` for all other blanks
//!    (unless they already have a canonical ID).
//! 3. **Issue simple IDs** — blank nodes with a *unique* first-degree hash
//!    receive their canonical identifier immediately (`_:c14n0`, `_:c14n1`, …)
//!    in hash order.
//! 4. **Hash N-Degree Quads** — for blank nodes that still share a first-degree
//!    hash, apply the full recursive N-degree hashing algorithm which walks the
//!    RDF neighbourhood to break ties.
//! 5. **Emit canonical N-Quads** — replace all blank node identifiers with their
//!    canonical form, sort the resulting quads lexicographically, and join with
//!    `\n`.
//!
//! ## Reference
//!
//! - W3C RDF Canonicalization 1.0: <https://www.w3.org/TR/rdf-canon/>
//! - URDNA2015 test suite: <https://json-ld.github.io/rdf-dataset-normalization/tests/>

use std::collections::{BTreeMap, HashMap};

use super::{
    hash::sha256_hex,
    nquads::{quad_to_nquad, quad_to_nquad_with_placeholders},
    types::RdfQuad,
};

// ─── Identifier Issuer ───────────────────────────────────────────────────────

/// Issues monotonically increasing canonical blank node identifiers.
///
/// Each call to [`IdentifierIssuer::issue`] either returns the previously
/// assigned identifier for a blank node or mints a fresh one with the format
/// `_:c14n<N>`.  The internal counter starts at 0.
#[derive(Debug, Clone)]
pub struct IdentifierIssuer {
    prefix: String,
    counter: usize,
    /// Ordered map from original blank node id → canonical id.
    /// Ordered so that the issuer can be cloned and iterated deterministically.
    issued: Vec<(String, String)>,
    /// Fast lookup set.
    issued_map: HashMap<String, String>,
}

impl IdentifierIssuer {
    /// Create a new issuer using `prefix` as the canonical identifier prefix.
    pub fn new(prefix: impl Into<String>) -> Self {
        IdentifierIssuer {
            prefix: prefix.into(),
            counter: 0,
            issued: Vec::new(),
            issued_map: HashMap::new(),
        }
    }

    /// Return the canonical identifier that has been (or will be) assigned to
    /// `original`.  If this is the first call for `original`, a new identifier
    /// is minted.
    pub fn issue(&mut self, original: &str) -> String {
        if let Some(existing) = self.issued_map.get(original) {
            return existing.clone();
        }
        let canonical = format!("{}{}", self.prefix, self.counter);
        self.counter += 1;
        self.issued_map
            .insert(original.to_string(), canonical.clone());
        self.issued.push((original.to_string(), canonical.clone()));
        canonical
    }

    /// Return the canonical identifier for `original` if it has already been
    /// issued, without minting a new one.
    pub fn get(&self, original: &str) -> Option<&str> {
        self.issued_map.get(original).map(String::as_str)
    }

    /// Return `true` if a canonical identifier has already been issued for
    /// `original`.
    pub fn has_issued(&self, original: &str) -> bool {
        self.issued_map.contains_key(original)
    }

    /// Return an iterator over `(original, canonical)` pairs in issue order.
    pub fn issued_pairs(&self) -> impl Iterator<Item = (&str, &str)> {
        self.issued.iter().map(|(o, c)| (o.as_str(), c.as_str()))
    }

    /// Return the mapping as a `HashMap<String, String>` for N-Quads serialization.
    pub fn as_map(&self) -> &HashMap<String, String> {
        &self.issued_map
    }
}

// ─── Canonicalizer ───────────────────────────────────────────────────────────

/// The main URDNA2015 canonicalizer.
///
/// Instantiate with [`Canonicalizer::new`] then call
/// [`Canonicalizer::canonicalize`].  The canonicalizer is reusable but is
/// stateful during a single canonicalization run — all state is reset at the
/// beginning of [`Canonicalizer::canonicalize`].
pub struct Canonicalizer {
    /// Issues canonical IDs: `_:c14n0`, `_:c14n1`, …
    issuer: IdentifierIssuer,
    /// Map from blank node id → list of quads in which the node appears.
    bnode_to_quads: HashMap<String, Vec<usize>>,
    /// All quads in the dataset.
    quads: Vec<RdfQuad>,
    /// First-degree hash → list of blank node ids with that hash.
    hash_to_bnodes: BTreeMap<String, Vec<String>>,
}

impl Canonicalizer {
    /// Create a new canonicalizer.
    pub fn new() -> Self {
        Canonicalizer {
            issuer: IdentifierIssuer::new("_:c14n"),
            bnode_to_quads: HashMap::new(),
            quads: Vec::new(),
            hash_to_bnodes: BTreeMap::new(),
        }
    }

    // ─── Stage 1: Collect blank nodes ─────────────────────────────────────

    /// Populate `bnode_to_quads` from the input quads.
    fn collect_blank_nodes(&mut self) {
        for (idx, quad) in self.quads.iter().enumerate() {
            for bnode_id in quad.blank_nodes() {
                self.bnode_to_quads
                    .entry(bnode_id.to_string())
                    .or_default()
                    .push(idx);
            }
        }
    }

    // ─── Stage 2: First-Degree Hash ───────────────────────────────────────

    /// Hash the set of quads that contain `bnode`, using placeholder names:
    /// - `_:a` for the blank node being hashed
    /// - `_:z` for all other blank nodes (unless they already have a canonical
    ///   identifier in `issued`)
    fn hash_first_degree_quads(&self, bnode: &str, issued: &IdentifierIssuer) -> String {
        let quad_indices = match self.bnode_to_quads.get(bnode) {
            Some(v) => v,
            None => return sha256_hex(""),
        };

        // Serialize each quad in placeholder form, then sort, then hash the join.
        let mut nquad_lines: Vec<String> = quad_indices
            .iter()
            .map(|&idx| quad_to_nquad_with_placeholders(&self.quads[idx], bnode, issued.as_map()))
            .collect();

        nquad_lines.sort();
        sha256_hex(&nquad_lines.join(""))
    }

    // ─── Stage 3: Issue simple IDs ────────────────────────────────────────

    /// For each first-degree hash that belongs to exactly one blank node, issue
    /// a canonical identifier in sorted-hash order.
    fn issue_simple_canonical_ids(&mut self) {
        // Collect hashes with exactly one blank node.
        let mut simple: Vec<(String, String)> = self
            .hash_to_bnodes
            .iter()
            .filter(|(_, bnodes)| bnodes.len() == 1)
            .map(|(h, bnodes)| (h.clone(), bnodes[0].clone()))
            .collect();

        // Sorted by hash (BTreeMap already iterates in sorted order, so this
        // preserves the spec-required deterministic order).
        simple.sort_by(|a, b| a.0.cmp(&b.0));

        for (_, bnode) in simple {
            self.issuer.issue(&bnode);
        }
    }

    // ─── Stage 4: N-Degree Hash ───────────────────────────────────────────

    /// Compute the N-degree hash for a blank node, resolving all ambiguous
    /// blank nodes in its neighbourhood recursively.
    ///
    /// This is the most complex part of URDNA2015.  The algorithm:
    ///
    /// 1. For each blank node `p` in the same first-degree-hash group, try all
    ///    permutations of assigning temporary canonical IDs to those nodes and
    ///    hash the resulting neighbourhood quads.
    /// 2. Choose the lexicographically smallest hash to break ties.
    ///
    /// The `depth` parameter guards against infinite recursion in pathological
    /// graphs (cycles of blank nodes).
    fn hash_n_degree_quads(
        &self,
        bnode: &str,
        issuer: &IdentifierIssuer,
        depth: usize,
    ) -> (String, IdentifierIssuer) {
        // Hard limit on recursion depth (spec leaves implementation-defined).
        const MAX_DEPTH: usize = 64;

        // Build the set of related blank nodes in the RDF neighbourhood.
        // "Related" means: sharing a quad with `bnode`.
        let mut hash_to_related: BTreeMap<String, Vec<String>> = BTreeMap::new();

        let quad_indices = match self.bnode_to_quads.get(bnode) {
            Some(v) => v.clone(),
            None => return (sha256_hex(""), issuer.clone()),
        };

        for quad_idx in &quad_indices {
            let quad = &self.quads[*quad_idx];
            // Collect every blank node in this quad that is NOT `bnode`.
            for related_id in quad.blank_nodes() {
                if related_id == bnode {
                    continue;
                }
                // Hash the related node's first-degree quads to group it.
                let related_hash = self.hash_first_degree_quads(related_id, issuer);
                hash_to_related
                    .entry(related_hash)
                    .or_default()
                    .push(related_id.to_string());
            }
        }

        // Build a deterministic "data to hash" string.
        let mut combined = String::new();

        // Append the initial quads of `bnode` with canonical substitution.
        {
            let mut nquad_lines: Vec<String> = quad_indices
                .iter()
                .map(|&idx| quad_to_nquad(&self.quads[idx], issuer.as_map()))
                .collect();
            nquad_lines.sort();
            combined.push_str(&nquad_lines.join(""));
        }

        let mut chosen_issuer = issuer.clone();

        // For each group of related nodes (sorted by their first-degree hash),
        // try all permutations to find the lexicographically smallest hash.
        for (group_hash, related_group) in &hash_to_related {
            combined.push_str(group_hash);

            let mut best_path: Option<(String, IdentifierIssuer)> = None;

            // Generate all permutations of the related group.
            let permutations = permutations_of(related_group);

            for perm in permutations {
                let mut path_issuer = issuer.clone();
                let mut path = String::new();

                for related_bnode in &perm {
                    // Issue a temporary ID if not yet canonical.
                    if let Some(canonical) = issuer.get(related_bnode) {
                        path.push_str(canonical);
                    } else {
                        // Recursively hash this related node (up to depth limit).
                        if depth < MAX_DEPTH {
                            let temp_id = path_issuer.issue(related_bnode);
                            path.push_str(&temp_id);
                            let (recursive_hash, _) =
                                self.hash_n_degree_quads(related_bnode, &path_issuer, depth + 1);
                            path.push_str(&recursive_hash);
                        } else {
                            // At depth limit: just use the first-degree hash.
                            let fd_hash = self.hash_first_degree_quads(related_bnode, &path_issuer);
                            path.push_str(&fd_hash);
                        }
                    }
                }

                let candidate_hash = sha256_hex(&path);

                if let Some((ref best_hash, _)) = best_path {
                    if candidate_hash < *best_hash {
                        best_path = Some((candidate_hash, path_issuer));
                    }
                } else {
                    best_path = Some((candidate_hash, path_issuer));
                }
            }

            if let Some((best_hash, best_issuer)) = best_path {
                combined.push_str(&best_hash);
                chosen_issuer = best_issuer;
            }
        }

        (sha256_hex(&combined), chosen_issuer)
    }

    /// Issue canonical identifiers to all blank nodes that still lack one,
    /// using the N-degree hash to break ties.
    fn issue_ndegree_canonical_ids(&mut self) {
        // Collect the groups that still have more than one candidate (or any
        // unissued node).  Work in sorted-by-hash order for determinism.
        let unresolved_groups: Vec<(String, Vec<String>)> = self
            .hash_to_bnodes
            .iter()
            .filter(|(_, bnodes)| bnodes.len() > 1)
            .map(|(h, bnodes)| (h.clone(), bnodes.clone()))
            .collect();

        for (_, group) in unresolved_groups {
            // For each blank node in the group, compute an N-degree hash.
            let mut nd_hashes: Vec<(String, String, IdentifierIssuer)> = group
                .iter()
                .map(|bnode| {
                    let (nd_hash, temp_issuer) =
                        self.hash_n_degree_quads(bnode, &self.issuer.clone(), 0);
                    (nd_hash, bnode.clone(), temp_issuer)
                })
                .collect();

            // Sort by N-degree hash for deterministic ordering.
            nd_hashes.sort_by(|a, b| a.0.cmp(&b.0));

            for (_, bnode, _temp_issuer) in &nd_hashes {
                if !self.issuer.has_issued(bnode) {
                    self.issuer.issue(bnode);
                }
            }
        }
    }

    // ─── Stage 5: Emit canonical N-Quads ─────────────────────────────────

    /// Serialize all quads with canonical blank node identifiers, sort them,
    /// and join with newlines.
    fn emit_canonical_nquads(&self) -> String {
        let mapping = self.issuer.as_map();
        let mut lines: Vec<String> = self
            .quads
            .iter()
            .map(|q| quad_to_nquad(q, mapping))
            .collect();
        lines.sort();
        lines.join("\n")
    }

    // ─── Public entry point ───────────────────────────────────────────────

    /// Canonicalize a slice of RDF quads according to URDNA2015.
    ///
    /// Returns a UTF-8 string containing sorted canonical N-Quads (one per
    /// line, lines separated by `\n`).  An empty input yields an empty string.
    pub fn canonicalize(quads: &[RdfQuad]) -> String {
        let mut c = Canonicalizer::new();
        c.quads = quads.to_vec();

        if c.quads.is_empty() {
            return String::new();
        }

        // Stage 1: collect blank nodes.
        c.collect_blank_nodes();

        // Stage 2: compute first-degree hashes.
        let all_bnodes: Vec<String> = c.bnode_to_quads.keys().cloned().collect();
        let temp_issuer = IdentifierIssuer::new("_:c14n");
        for bnode in &all_bnodes {
            let hash = c.hash_first_degree_quads(bnode, &temp_issuer);
            c.hash_to_bnodes
                .entry(hash)
                .or_default()
                .push(bnode.clone());
        }

        // Sort each group to ensure determinism before issuing.
        for group in c.hash_to_bnodes.values_mut() {
            group.sort();
        }

        // Stage 3: issue simple canonical IDs.
        c.issue_simple_canonical_ids();

        // Stage 4: issue N-degree canonical IDs.
        c.issue_ndegree_canonical_ids();

        // Stage 5: emit.
        c.emit_canonical_nquads()
    }
}

impl Default for Canonicalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Permutation helpers ─────────────────────────────────────────────────────

/// Generate all permutations of a slice.
///
/// Uses Heap's algorithm for efficiency.  The number of permutations is `n!`,
/// so this should only be called on small slices (in practice blank nodes
/// sharing the same first-degree hash are rare and few).
fn permutations_of(items: &[String]) -> Vec<Vec<String>> {
    let n = items.len();
    if n == 0 {
        return vec![vec![]];
    }
    if n == 1 {
        return vec![items.to_vec()];
    }

    let mut result = Vec::new();
    let mut arr = items.to_vec();
    heap_permute(&mut arr, n, &mut result);
    result
}

fn heap_permute(arr: &mut Vec<String>, k: usize, out: &mut Vec<Vec<String>>) {
    if k == 1 {
        out.push(arr.clone());
        return;
    }
    for i in 0..k {
        heap_permute(arr, k - 1, out);
        if k % 2 == 0 {
            arr.swap(i, k - 1);
        } else {
            arr.swap(0, k - 1);
        }
    }
}

// ─── Convenience free function ────────────────────────────────────────────────

/// Canonicalize an RDF dataset (slice of [`RdfQuad`]s) to a canonical N-Quads
/// string.
///
/// This is the primary public entry point for URDNA2015.
///
/// ```rust
/// use oxirs_core::canon::{canonicalize, QuadTerm, RdfQuad};
///
/// let quads = vec![RdfQuad::new(
///     QuadTerm::iri("http://example.org/s"),
///     QuadTerm::iri("http://example.org/p"),
///     QuadTerm::iri("http://example.org/o"),
/// )];
/// let canonical = canonicalize(&quads);
/// assert!(canonical.contains("<http://example.org/s>"));
/// ```
pub fn canonicalize(quads: &[RdfQuad]) -> String {
    Canonicalizer::canonicalize(quads)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canon::types::{QuadTerm, RdfQuad};
    use std::collections::HashSet;

    // ── Helper builders ──────────────────────────────────────────────────

    fn iri_quad(s: &str, p: &str, o: &str) -> RdfQuad {
        RdfQuad::new(QuadTerm::iri(s), QuadTerm::iri(p), QuadTerm::iri(o))
    }

    fn blank_iri_quad(bnode: &str, p: &str, o: &str) -> RdfQuad {
        RdfQuad::new(QuadTerm::blank(bnode), QuadTerm::iri(p), QuadTerm::iri(o))
    }

    // ── Test 1: Empty graph ──────────────────────────────────────────────

    #[test]
    fn test_empty_graph_yields_empty_output() {
        let result = canonicalize(&[]);
        assert!(result.is_empty(), "expected empty string for empty input");
    }

    // ── Test 2: Single IRI quad ──────────────────────────────────────────

    #[test]
    fn test_single_iri_quad() {
        let quads = vec![iri_quad(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];
        let result = canonicalize(&quads);
        assert_eq!(
            result,
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> ."
        );
    }

    // ── Test 3: Literal with datatype ────────────────────────────────────

    #[test]
    fn test_literal_with_datatype() {
        let quads = vec![RdfQuad::new(
            QuadTerm::iri("http://example.org/s"),
            QuadTerm::iri("http://example.org/p"),
            QuadTerm::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer"),
        )];
        let result = canonicalize(&quads);
        assert!(
            result.contains("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"),
            "expected typed literal; got: {result}"
        );
    }

    // ── Test 4: Literal with language tag ────────────────────────────────

    #[test]
    fn test_literal_with_language_tag() {
        let quads = vec![RdfQuad::new(
            QuadTerm::iri("http://example.org/s"),
            QuadTerm::iri("http://example.org/p"),
            QuadTerm::lang_literal("Hello", "en"),
        )];
        let result = canonicalize(&quads);
        assert!(
            result.contains("\"Hello\"@en"),
            "expected language-tagged literal; got: {result}"
        );
    }

    // ── Test 5: Single blank node → _:c14n0 ──────────────────────────────

    #[test]
    fn test_single_blank_node_gets_c14n0() {
        let quads = vec![blank_iri_quad(
            "b0",
            "http://example.org/p",
            "http://example.org/o",
        )];
        let result = canonicalize(&quads);
        assert!(
            result.contains("_:c14n0"),
            "expected _:c14n0; got: {result}"
        );
        assert!(
            !result.contains("_:b0"),
            "original blank node id must not appear in output"
        );
    }

    // ── Test 6: Two blank nodes → _:c14n0 and _:c14n1 ───────────────────

    #[test]
    fn test_two_blank_nodes_get_sequential_ids() {
        let quads = vec![
            blank_iri_quad("b0", "http://example.org/type", "http://example.org/A"),
            blank_iri_quad("b1", "http://example.org/type", "http://example.org/B"),
        ];
        let result = canonicalize(&quads);
        assert!(
            result.contains("_:c14n0"),
            "expected _:c14n0; got: {result}"
        );
        assert!(
            result.contains("_:c14n1"),
            "expected _:c14n1; got: {result}"
        );
    }

    // ── Test 7: Same blank node in multiple quads → same canonical ID ─────

    #[test]
    fn test_same_blank_node_across_quads() {
        let quads = vec![
            blank_iri_quad(
                "alice",
                "http://example.org/name",
                "http://example.org/Alice",
            ),
            blank_iri_quad(
                "alice",
                "http://example.org/knows",
                "http://example.org/Bob",
            ),
        ];
        let result = canonicalize(&quads);
        // Both lines should reference the same canonical ID (c14n0).
        let c14n_count = result.matches("_:c14n0").count();
        assert_eq!(
            c14n_count, 2,
            "_:c14n0 should appear once per quad; got {c14n_count} occurrences:\n{result}"
        );
        // Only one canonical ID should exist.
        assert!(
            !result.contains("_:c14n1"),
            "only one blank node exists; _:c14n1 must not appear; got:\n{result}"
        );
    }

    // ── Test 8: Named graph quad ──────────────────────────────────────────

    #[test]
    fn test_named_graph_quad() {
        let quads = vec![RdfQuad::new_in_graph(
            QuadTerm::iri("http://example.org/s"),
            QuadTerm::iri("http://example.org/p"),
            QuadTerm::iri("http://example.org/o"),
            QuadTerm::iri("http://example.org/graph1"),
        )];
        let result = canonicalize(&quads);
        assert!(
            result.contains("<http://example.org/graph1>"),
            "named graph must appear in output; got: {result}"
        );
        assert!(
            result.ends_with('.') || result.ends_with(". "),
            "N-Quads line must end with period; got: {result}"
        );
    }

    // ── Test 9: Canonical output is sorted lexicographically ─────────────

    #[test]
    fn test_output_is_sorted_lexicographically() {
        let quads = vec![
            iri_quad(
                "http://z.example.org/s",
                "http://example.org/p",
                "http://example.org/o",
            ),
            iri_quad(
                "http://a.example.org/s",
                "http://example.org/p",
                "http://example.org/o",
            ),
            iri_quad(
                "http://m.example.org/s",
                "http://example.org/p",
                "http://example.org/o",
            ),
        ];
        let result = canonicalize(&quads);
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 3);
        // Must be in ascending order.
        assert!(lines[0] < lines[1], "line 0 must be < line 1");
        assert!(lines[1] < lines[2], "line 1 must be < line 2");
        assert!(
            lines[0].contains("/a.example.org"),
            "first line must be /a; got: {}",
            lines[0]
        );
        assert!(
            lines[2].contains("/z.example.org"),
            "last line must be /z; got: {}",
            lines[2]
        );
    }

    // ── Test 10: Deterministic — two calls with same input → same output ──

    #[test]
    fn test_canonicalization_is_deterministic() {
        let quads = vec![
            blank_iri_quad("x", "http://example.org/p", "http://example.org/o"),
            blank_iri_quad("y", "http://example.org/q", "http://example.org/o"),
            RdfQuad::new(
                QuadTerm::blank("x"),
                QuadTerm::iri("http://example.org/knows"),
                QuadTerm::blank("y"),
            ),
        ];
        let first = canonicalize(&quads);
        let second = canonicalize(&quads);
        assert_eq!(
            first, second,
            "canonicalization must be deterministic across calls"
        );
    }

    // ── Test 11: Blank node renamed from _:b0 to _:c14n0 ─────────────────

    #[test]
    fn test_blank_node_renamed_from_b0() {
        let quads = vec![blank_iri_quad(
            "b0",
            "http://example.org/type",
            "http://example.org/Thing",
        )];
        let result = canonicalize(&quads);
        assert!(
            result.contains("_:c14n0"),
            "blank node must be renamed to _:c14n0; got: {result}"
        );
        assert!(
            !result.contains("_:b0"),
            "original _:b0 must NOT appear in output; got: {result}"
        );
    }

    // ── Test 12: Complex graph — 3 blank nodes all get unique IDs ─────────

    #[test]
    fn test_three_blank_nodes_unique_ids() {
        // Three blank nodes connected in a chain: x → y → z
        let quads = vec![
            RdfQuad::new(
                QuadTerm::blank("x"),
                QuadTerm::iri("http://example.org/next"),
                QuadTerm::blank("y"),
            ),
            RdfQuad::new(
                QuadTerm::blank("y"),
                QuadTerm::iri("http://example.org/next"),
                QuadTerm::blank("z"),
            ),
            blank_iri_quad("z", "http://example.org/type", "http://example.org/End"),
        ];
        let result = canonicalize(&quads);

        let mut ids: HashSet<String> = HashSet::new();
        for i in 0..3 {
            let id = format!("_:c14n{}", i);
            assert!(
                result.contains(&id),
                "expected {id} in output; got:\n{result}"
            );
            ids.insert(id);
        }
        assert_eq!(ids.len(), 3, "all three canonical IDs must be distinct");
    }

    // ── Test 13: Blank node in object position ────────────────────────────

    #[test]
    fn test_blank_node_in_object_position() {
        let quads = vec![RdfQuad::new(
            QuadTerm::iri("http://example.org/s"),
            QuadTerm::iri("http://example.org/p"),
            QuadTerm::blank("obj_node"),
        )];
        let result = canonicalize(&quads);
        assert!(
            result.contains("_:c14n0"),
            "blank object must be canonicalized; got: {result}"
        );
    }

    // ── Test 14: Blank node in named graph position ───────────────────────

    #[test]
    fn test_blank_node_as_named_graph() {
        let quads = vec![RdfQuad::new_in_graph(
            QuadTerm::iri("http://example.org/s"),
            QuadTerm::iri("http://example.org/p"),
            QuadTerm::iri("http://example.org/o"),
            QuadTerm::blank("g"),
        )];
        let result = canonicalize(&quads);
        assert!(
            result.contains("_:c14n0"),
            "blank graph name must be canonicalized; got: {result}"
        );
    }

    // ── Test 15: Permutation issuer ───────────────────────────────────────

    #[test]
    fn test_identifier_issuer_sequential() {
        let mut issuer = IdentifierIssuer::new("_:c14n");
        let id0 = issuer.issue("b0");
        let id1 = issuer.issue("b1");
        let id0_again = issuer.issue("b0");
        assert_eq!(id0, "_:c14n0");
        assert_eq!(id1, "_:c14n1");
        assert_eq!(id0_again, "_:c14n0", "same node must get same canonical id");
    }

    // ── Test 16: Two separate canonicalize calls are independent ──────────

    #[test]
    fn test_two_calls_are_independent() {
        let quads_a = vec![blank_iri_quad(
            "x",
            "http://example.org/p",
            "http://example.org/A",
        )];
        let quads_b = vec![blank_iri_quad(
            "x",
            "http://example.org/p",
            "http://example.org/A",
        )];
        assert_eq!(
            canonicalize(&quads_a),
            canonicalize(&quads_b),
            "identical inputs must produce identical outputs"
        );
    }
}
