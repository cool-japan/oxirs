//! Frozen, mmap-able snapshot format for sub-second cold starts of a read-only
//! [`MemoryStorage`](super::storage::MemoryStorage).
//!
//! ## Why
//!
//! The durable backend loads its dataset by re-parsing `data.nq` line by line on
//! every process start — for a 1.35M-quad / 158 MB dataset that is ~13.6 s of
//! N-Quads parse + intern + four `BTreeSet` inserts per quad, paid again on every
//! container boot of a read-only deployment. A *snapshot* trades that repeated
//! parse for a one-time offline build: the interned term dictionaries and the
//! four sorted permutation indexes are serialized to a single file that the
//! loader `mmap`s and rebuilds from **without any line parsing or sorting**.
//!
//! ## Format (`snapshot.oxsnap`, all little-endian)
//!
//! ```text
//! Header (40 bytes)
//!   0  ..8    magic  = b"OXRSSNAP"
//!   8  ..12   version: u32                (must equal FORMAT_VERSION)
//!   12 ..13   endianness: u8              (1 = little; loader rejects others)
//!   13 ..14   flags: u8                   (reserved, 0)
//!   14 ..16   reserved: u16               (0)
//!   16 ..24   quad_count: u64
//!   24 ..32   source_len: u64             (data.nq length at build time; a
//!                                          cheap staleness guard, 0 = unknown)
//!   32 ..40   reserved2: u64              (0)
//! Directory (192 bytes) — absolute file offsets
//!   4 dict entries  (subjects, predicates, objects, graphs), 32 bytes each:
//!       pool_off: u64, pool_len: u64, offsets_off: u64, term_count: u64
//!   4 index entries (spog, posg, ospg, gspo), 16 bytes each:
//!       array_off: u64, tuple_count: u64
//! Sections (each 8-byte aligned, in directory order)
//!   per dict: string-pool bytes, then (term_count+1) u64 pool offsets
//!   per index: tuple_count * [u32;4]  (16 bytes/tuple), little-endian
//! ```
//!
//! ### String pool entries (per term, self-describing, tag byte first)
//!
//! Terms are stored **sorted by their encoded bytes** so a column's pool is in
//! dictionary order (grouping like-typed terms and lexicographically ordering
//! IRIs/literals — ready for a future front-coding pass) and the loader can
//! assign id = sorted rank deterministically. Encoding is a direct, parse-free
//! inverse of the model constructors:
//!
//! ```text
//!   0x01 NamedNode     : iri bytes
//!   0x02 BlankNode     : id bytes (no _: prefix)
//!   0x03 Literal simple: value bytes
//!   0x04 Literal lang  : value | language | u32 value_len (tail)
//!   0x05 Literal typed : value | datatype-iri | u32 value_len (tail)
//!   0x06 Variable      : name bytes
//!   0x07 DefaultGraph   : (empty)
//!   0x08 Literal dir   : value | language | dir_byte | u32 value_len (tail)  [rdf-12]
//! ```
//!
//! The value length lives at the *tail* so the leading bytes stay
//! value-lexicographic (front-coding friendly); the remaining split is recovered
//! from the entry's total length (known from the offset table). RDF-star quoted
//! triples are not yet representable and cause [`write_snapshot`] to return an
//! error, so the caller falls back to the `data.nq` load — never a silent loss.
//!
//! ## Determinism & safety
//!
//! Output is a pure function of the quad *set*: ids are re-derived from the
//! sorted term order (independent of the source store's insertion order), the
//! permutations are regenerated and sorted, and every reserved/pad byte is zero —
//! so identical data yields identical bytes. The loader validates the magic,
//! version, endianness, and every section bound before use, and re-derives all
//! per-id reference counts from the SPOG array; any structural inconsistency, a
//! version mismatch, or a `source_len` that no longer matches the current
//! `data.nq` yields an `Err`, and the open path treats that as "fall back to the
//! plain `data.nq` load" rather than panicking.

use super::dictionary::ColumnDictionary;
use super::storage::MemoryStorage;
use crate::model::{
    BlankNode, GraphName, Literal, NamedNode, Object, Predicate, Subject, Variable,
};
use crate::{OxirsError, Result};
use memmap2::Mmap;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Default snapshot file name, a sibling of `data.nq` in a dataset directory.
pub const SNAPSHOT_FILE_NAME: &str = "snapshot.oxsnap";

/// File magic; also the informal format name.
const MAGIC: &[u8; 8] = b"OXRSSNAP";
/// On-disk format version. A loader refuses any other value (safe fallback).
const FORMAT_VERSION: u32 = 1;
/// Endianness marker written into the header (`1` = little-endian).
const ENDIAN_LE: u8 = 1;

const HEADER_LEN: u64 = 40;
const DIRECTORY_LEN: u64 = 4 * 32 + 4 * 16; // four dict entries + four index entries
const DATA_START: u64 = HEADER_LEN + DIRECTORY_LEN; // 232, already 8-aligned

// Term encoding tags.
const TAG_NAMED: u8 = 0x01;
const TAG_BLANK: u8 = 0x02;
const TAG_LIT_SIMPLE: u8 = 0x03;
const TAG_LIT_LANG: u8 = 0x04;
const TAG_LIT_TYPED: u8 = 0x05;
const TAG_VARIABLE: u8 = 0x06;
const TAG_DEFAULT_GRAPH: u8 = 0x07;
#[cfg(feature = "rdf-12")]
const TAG_LIT_DIR: u8 = 0x08;

/// Round `offset` up to the next multiple of 8 (section alignment).
#[inline]
fn align8(offset: u64) -> u64 {
    (offset + 7) & !7
}

// ---------------------------------------------------------------------------
// Term encoding (build side)
// ---------------------------------------------------------------------------

fn encode_named(iri: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + iri.len());
    out.push(TAG_NAMED);
    out.extend_from_slice(iri.as_bytes());
    out
}

fn encode_blank(id: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + id.len());
    out.push(TAG_BLANK);
    out.extend_from_slice(id.as_bytes());
    out
}

fn encode_variable(name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + name.len());
    out.push(TAG_VARIABLE);
    out.extend_from_slice(name.as_bytes());
    out
}

fn encode_literal(literal: &Literal) -> Vec<u8> {
    let value = literal.value();

    #[cfg(feature = "rdf-12")]
    if let Some(direction) = literal.direction() {
        use crate::model::literal::BaseDirection;
        let language = literal.language().unwrap_or("");
        let mut out = Vec::with_capacity(1 + value.len() + language.len() + 1 + 4);
        out.push(TAG_LIT_DIR);
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(language.as_bytes());
        out.push(match direction {
            BaseDirection::Ltr => 0,
            BaseDirection::Rtl => 1,
        });
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        return out;
    }

    if let Some(language) = literal.language() {
        let mut out = Vec::with_capacity(1 + value.len() + language.len() + 4);
        out.push(TAG_LIT_LANG);
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(language.as_bytes());
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        return out;
    }

    if literal.is_typed() {
        let datatype = literal.datatype();
        let dt = datatype.as_str();
        let mut out = Vec::with_capacity(1 + value.len() + dt.len() + 4);
        out.push(TAG_LIT_TYPED);
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(dt.as_bytes());
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        return out;
    }

    let mut out = Vec::with_capacity(1 + value.len());
    out.push(TAG_LIT_SIMPLE);
    out.extend_from_slice(value.as_bytes());
    out
}

fn quoted_triple_unsupported(column: &str) -> OxirsError {
    OxirsError::Store(format!(
        "snapshot: RDF-star quoted triples in the {column} column are not yet \
         representable in the frozen snapshot format; skipping snapshot (the \
         plain data.nq load remains available)"
    ))
}

fn encode_subject(subject: &Subject) -> Result<Vec<u8>> {
    match subject {
        Subject::NamedNode(n) => Ok(encode_named(n.as_str())),
        Subject::BlankNode(b) => Ok(encode_blank(b.id())),
        Subject::Variable(v) => Ok(encode_variable(v.name())),
        Subject::QuotedTriple(_) => Err(quoted_triple_unsupported("subject")),
    }
}

fn encode_predicate(predicate: &Predicate) -> Result<Vec<u8>> {
    match predicate {
        Predicate::NamedNode(n) => Ok(encode_named(n.as_str())),
        Predicate::Variable(v) => Ok(encode_variable(v.name())),
    }
}

fn encode_object(object: &Object) -> Result<Vec<u8>> {
    match object {
        Object::NamedNode(n) => Ok(encode_named(n.as_str())),
        Object::BlankNode(b) => Ok(encode_blank(b.id())),
        Object::Literal(l) => Ok(encode_literal(l)),
        Object::Variable(v) => Ok(encode_variable(v.name())),
        Object::QuotedTriple(_) => Err(quoted_triple_unsupported("object")),
    }
}

fn encode_graph(graph: &GraphName) -> Result<Vec<u8>> {
    match graph {
        GraphName::NamedNode(n) => Ok(encode_named(n.as_str())),
        GraphName::BlankNode(b) => Ok(encode_blank(b.id())),
        GraphName::Variable(v) => Ok(encode_variable(v.name())),
        GraphName::DefaultGraph => Ok(vec![TAG_DEFAULT_GRAPH]),
    }
}

// ---------------------------------------------------------------------------
// Term decoding (load side)
// ---------------------------------------------------------------------------

fn snapshot_err(msg: impl Into<String>) -> OxirsError {
    OxirsError::Store(format!("snapshot: {}", msg.into()))
}

fn decode_str(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|e| snapshot_err(format!("invalid UTF-8 in term: {e}")))
}

/// Split a length-tailed literal body (`value | rest | u32 value_len`) into its
/// `value` and `rest` byte slices. `payload` is the entry bytes *after* the tag.
fn split_length_tailed(payload: &[u8]) -> Result<(&[u8], &[u8])> {
    if payload.len() < 4 {
        return Err(snapshot_err("literal entry too short for length tail"));
    }
    let (body, tail) = payload.split_at(payload.len() - 4);
    let value_len = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as usize;
    if value_len > body.len() {
        return Err(snapshot_err("literal value length exceeds entry"));
    }
    Ok(body.split_at(value_len))
}

fn decode_literal(entry: &[u8]) -> Result<Literal> {
    let (&tag, payload) = entry
        .split_first()
        .ok_or_else(|| snapshot_err("empty literal entry"))?;
    match tag {
        TAG_LIT_SIMPLE => Ok(Literal::new_simple_literal(decode_str(payload)?)),
        TAG_LIT_LANG => {
            let (value, language) = split_length_tailed(payload)?;
            Ok(Literal::new_language_tagged_literal_unchecked(
                decode_str(value)?,
                decode_str(language)?,
            ))
        }
        TAG_LIT_TYPED => {
            let (value, datatype) = split_length_tailed(payload)?;
            Ok(Literal::new_typed_literal(
                decode_str(value)?.to_string(),
                NamedNode::new_unchecked(decode_str(datatype)?),
            ))
        }
        #[cfg(feature = "rdf-12")]
        TAG_LIT_DIR => {
            use crate::model::literal::BaseDirection;
            let (value, rest) = split_length_tailed(payload)?;
            let (&dir_byte, language) = rest
                .split_last()
                .ok_or_else(|| snapshot_err("directional literal missing direction byte"))?;
            let direction = match dir_byte {
                0 => BaseDirection::Ltr,
                1 => BaseDirection::Rtl,
                other => {
                    return Err(snapshot_err(format!("invalid direction byte {other}")));
                }
            };
            Ok(Literal::new_directional_language_tagged_literal_unchecked(
                decode_str(value)?,
                decode_str(language)?,
                direction,
            ))
        }
        other => Err(snapshot_err(format!("invalid literal tag {other:#04x}"))),
    }
}

fn decode_subject(entry: &[u8]) -> Result<Subject> {
    let (&tag, payload) = entry
        .split_first()
        .ok_or_else(|| snapshot_err("empty subject entry"))?;
    match tag {
        TAG_NAMED => Ok(Subject::NamedNode(NamedNode::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_BLANK => Ok(Subject::BlankNode(BlankNode::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_VARIABLE => Ok(Subject::Variable(Variable::new_unchecked(decode_str(
            payload,
        )?))),
        other => Err(snapshot_err(format!("invalid subject tag {other:#04x}"))),
    }
}

fn decode_predicate(entry: &[u8]) -> Result<Predicate> {
    let (&tag, payload) = entry
        .split_first()
        .ok_or_else(|| snapshot_err("empty predicate entry"))?;
    match tag {
        TAG_NAMED => Ok(Predicate::NamedNode(NamedNode::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_VARIABLE => Ok(Predicate::Variable(Variable::new_unchecked(decode_str(
            payload,
        )?))),
        other => Err(snapshot_err(format!("invalid predicate tag {other:#04x}"))),
    }
}

fn decode_object(entry: &[u8]) -> Result<Object> {
    let (&tag, payload) = entry
        .split_first()
        .ok_or_else(|| snapshot_err("empty object entry"))?;
    match tag {
        TAG_NAMED => Ok(Object::NamedNode(NamedNode::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_BLANK => Ok(Object::BlankNode(BlankNode::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_VARIABLE => Ok(Object::Variable(Variable::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_LIT_SIMPLE | TAG_LIT_LANG | TAG_LIT_TYPED => {
            Ok(Object::Literal(decode_literal(entry)?))
        }
        #[cfg(feature = "rdf-12")]
        TAG_LIT_DIR => Ok(Object::Literal(decode_literal(entry)?)),
        other => Err(snapshot_err(format!("invalid object tag {other:#04x}"))),
    }
}

fn decode_graph(entry: &[u8]) -> Result<GraphName> {
    let (&tag, payload) = entry
        .split_first()
        .ok_or_else(|| snapshot_err("empty graph entry"))?;
    match tag {
        TAG_NAMED => Ok(GraphName::NamedNode(NamedNode::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_BLANK => Ok(GraphName::BlankNode(BlankNode::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_VARIABLE => Ok(GraphName::Variable(Variable::new_unchecked(decode_str(
            payload,
        )?))),
        TAG_DEFAULT_GRAPH => Ok(GraphName::DefaultGraph),
        other => Err(snapshot_err(format!("invalid graph tag {other:#04x}"))),
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// A column serialized for the snapshot: the concatenated sorted pool bytes, the
/// `(n+1)` pool offsets, and the `old_id -> new_id` remap (indexed by old id;
/// tombstoned slots hold `u32::MAX` and are never referenced by a live tuple).
struct BuiltColumn {
    pool: Vec<u8>,
    offsets: Vec<u64>,
    remap: Vec<u32>,
    term_count: u64,
}

/// Encode a column's live terms, sort them by encoded bytes (dictionary order),
/// and assign each the id = its sorted rank, recording the remap from the source
/// store's ids.
fn build_column<T, F>(dict: &ColumnDictionary<T>, slot_len: usize, encode: F) -> Result<BuiltColumn>
where
    T: Clone + Eq + std::hash::Hash,
    F: Fn(&T) -> Result<Vec<u8>>,
{
    let mut entries: Vec<(u32, Vec<u8>)> = Vec::with_capacity(dict.live_len());
    for (old_id, term) in dict.iter_live_slots() {
        entries.push((old_id, encode(term)?));
    }
    // Encoded bytes are injective across distinct terms, so this is a total order
    // and the resulting layout is independent of the source insertion order.
    entries.sort_unstable_by(|a, b| a.1.cmp(&b.1));

    let term_count = entries.len();
    let mut remap = vec![u32::MAX; slot_len];
    let mut pool = Vec::new();
    let mut offsets = Vec::with_capacity(term_count + 1);
    offsets.push(0u64);
    for (new_id, (old_id, encoded)) in entries.iter().enumerate() {
        remap[*old_id as usize] = new_id as u32;
        pool.extend_from_slice(encoded);
        offsets.push(pool.len() as u64);
    }
    Ok(BuiltColumn {
        pool,
        offsets,
        remap,
        term_count: term_count as u64,
    })
}

/// Serialize `storage` to a frozen snapshot at `path`.
///
/// Deterministic: the same quad set always produces byte-identical output. Fails
/// (leaving no partial file behind other than a `.tmp`) if the store contains
/// RDF-star quoted triples, which the format does not yet represent.
pub fn write_snapshot(storage: &MemoryStorage, path: &Path) -> Result<()> {
    let subjects = build_column(
        storage.subjects_dict(),
        storage.subjects_dict().slot_len(),
        encode_subject,
    )?;
    let predicates = build_column(
        storage.predicates_dict(),
        storage.predicates_dict().slot_len(),
        encode_predicate,
    )?;
    let objects = build_column(
        storage.objects_dict(),
        storage.objects_dict().slot_len(),
        encode_object,
    )?;
    let graphs = build_column(
        storage.graphs_dict(),
        storage.graphs_dict().slot_len(),
        encode_graph,
    )?;

    // Regenerate the four permutations from SPOG in the new (sorted) id space,
    // then sort each in its own permutation order so the loader builds its
    // BTreeSets from already-ordered input.
    let quad_count = storage.spog_index().len();
    let mut canonical: Vec<[u32; 4]> = Vec::with_capacity(quad_count);
    for tuple in storage.spog_index() {
        let s = remap_id(&subjects.remap, tuple[0], "subject")?;
        let p = remap_id(&predicates.remap, tuple[1], "predicate")?;
        let o = remap_id(&objects.remap, tuple[2], "object")?;
        let g = remap_id(&graphs.remap, tuple[3], "graph")?;
        canonical.push([s, p, o, g]);
    }

    let mut spog = canonical.clone();
    spog.sort_unstable();
    let mut posg: Vec<[u32; 4]> = canonical.iter().map(|q| [q[1], q[2], q[0], q[3]]).collect();
    posg.sort_unstable();
    let mut ospg: Vec<[u32; 4]> = canonical.iter().map(|q| [q[2], q[0], q[1], q[3]]).collect();
    ospg.sort_unstable();
    let mut gspo: Vec<[u32; 4]> = canonical.iter().map(|q| [q[3], q[0], q[1], q[2]]).collect();
    gspo.sort_unstable();
    drop(canonical);

    let source_len = source_data_len(path);

    // Lay out sections after the header+directory, each 8-byte aligned, and record
    // absolute offsets for the directory.
    let dict_cols = [&subjects, &predicates, &objects, &graphs];
    let indexes = [&spog, &posg, &ospg, &gspo];

    let mut cursor = DATA_START;
    let mut dict_layout: Vec<(u64, u64, u64)> = Vec::with_capacity(4); // pool_off, pool_len, offsets_off
    for col in dict_cols {
        cursor = align8(cursor);
        let pool_off = cursor;
        let pool_len = col.pool.len() as u64;
        cursor += pool_len;
        cursor = align8(cursor);
        let offsets_off = cursor;
        cursor += (col.offsets.len() as u64) * 8;
        dict_layout.push((pool_off, pool_len, offsets_off));
    }
    let mut index_layout: Vec<u64> = Vec::with_capacity(4); // array_off
    for arr in indexes {
        cursor = align8(cursor);
        index_layout.push(cursor);
        cursor += (arr.len() as u64) * 16;
    }

    // Write to a temp file then atomically rename, so a reader never sees a
    // half-written snapshot.
    let tmp_path = path.with_extension("oxsnap.tmp");
    {
        let file = File::create(&tmp_path)
            .map_err(|e| OxirsError::Io(format!("Failed to create {}: {e}", tmp_path.display())))?;
        let mut writer = BufWriter::new(file);

        // Header.
        let mut header = Vec::with_capacity(HEADER_LEN as usize);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        header.push(ENDIAN_LE);
        header.push(0); // flags
        header.extend_from_slice(&0u16.to_le_bytes()); // reserved
        header.extend_from_slice(&(quad_count as u64).to_le_bytes());
        header.extend_from_slice(&source_len.to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes()); // reserved2
        debug_assert_eq!(header.len() as u64, HEADER_LEN);

        // Directory.
        let mut directory = Vec::with_capacity(DIRECTORY_LEN as usize);
        for (i, col) in dict_cols.iter().enumerate() {
            let (pool_off, pool_len, offsets_off) = dict_layout[i];
            directory.extend_from_slice(&pool_off.to_le_bytes());
            directory.extend_from_slice(&pool_len.to_le_bytes());
            directory.extend_from_slice(&offsets_off.to_le_bytes());
            directory.extend_from_slice(&col.term_count.to_le_bytes());
        }
        for (i, arr) in indexes.iter().enumerate() {
            directory.extend_from_slice(&index_layout[i].to_le_bytes());
            directory.extend_from_slice(&(arr.len() as u64).to_le_bytes());
        }
        debug_assert_eq!(directory.len() as u64, DIRECTORY_LEN);

        writer
            .write_all(&header)
            .and_then(|_| writer.write_all(&directory))
            .map_err(write_io_err(&tmp_path))?;
        let mut written = DATA_START;

        // Dict sections.
        for (i, col) in dict_cols.iter().enumerate() {
            let (pool_off, _pool_len, offsets_off) = dict_layout[i];
            pad_to(&mut writer, &mut written, pool_off, &tmp_path)?;
            writer
                .write_all(&col.pool)
                .map_err(write_io_err(&tmp_path))?;
            written += col.pool.len() as u64;
            pad_to(&mut writer, &mut written, offsets_off, &tmp_path)?;
            for &off in &col.offsets {
                writer
                    .write_all(&off.to_le_bytes())
                    .map_err(write_io_err(&tmp_path))?;
            }
            written += (col.offsets.len() as u64) * 8;
        }
        // Index sections.
        for (i, arr) in indexes.iter().enumerate() {
            pad_to(&mut writer, &mut written, index_layout[i], &tmp_path)?;
            for tuple in arr.iter() {
                let mut buf = [0u8; 16];
                buf[0..4].copy_from_slice(&tuple[0].to_le_bytes());
                buf[4..8].copy_from_slice(&tuple[1].to_le_bytes());
                buf[8..12].copy_from_slice(&tuple[2].to_le_bytes());
                buf[12..16].copy_from_slice(&tuple[3].to_le_bytes());
                writer.write_all(&buf).map_err(write_io_err(&tmp_path))?;
            }
            written += (arr.len() as u64) * 16;
        }

        writer.flush().map_err(write_io_err(&tmp_path))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(write_io_err(&tmp_path))?;
    }

    std::fs::rename(&tmp_path, path).map_err(|e| {
        OxirsError::Io(format!(
            "Failed to atomically place snapshot {}: {e}",
            path.display()
        ))
    })?;
    Ok(())
}

fn remap_id(remap: &[u32], old_id: u32, column: &str) -> Result<u32> {
    match remap.get(old_id as usize) {
        Some(&new_id) if new_id != u32::MAX => Ok(new_id),
        _ => Err(snapshot_err(format!(
            "internal: {column} id {old_id} has no sorted-rank mapping"
        ))),
    }
}

fn write_io_err(path: &Path) -> impl Fn(std::io::Error) -> OxirsError + '_ {
    move |e| OxirsError::Io(format!("Failed to write {}: {e}", path.display()))
}

/// Emit zero padding until the writer's logical position reaches `target`.
fn pad_to(writer: &mut BufWriter<File>, written: &mut u64, target: u64, path: &Path) -> Result<()> {
    while *written < target {
        writer.write_all(&[0u8]).map_err(write_io_err(path))?;
        *written += 1;
    }
    Ok(())
}

/// Best-effort length of the sibling `data.nq` for the staleness guard. `0` when
/// there is no sibling data file (the snapshot is then trusted unconditionally).
fn source_data_len(snapshot_path: &Path) -> u64 {
    let data_file = match snapshot_path.parent() {
        Some(dir) => dir.join("data.nq"),
        None => return 0,
    };
    std::fs::metadata(data_file).map(|m| m.len()).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Little-endian `u64` at `offset`, bounds-checked against `data`.
fn read_u64(data: &[u8], offset: u64) -> Result<u64> {
    let start = offset as usize;
    let end = start
        .checked_add(8)
        .ok_or_else(|| snapshot_err("offset overflow"))?;
    let slice = data
        .get(start..end)
        .ok_or_else(|| snapshot_err("truncated (u64 out of bounds)"))?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Ok(u64::from_le_bytes(buf))
}

/// Directory description of one column.
struct DictDir {
    pool_off: u64,
    pool_len: u64,
    offsets_off: u64,
    term_count: u64,
}

/// Directory description of one permutation index.
struct IndexDir {
    array_off: u64,
    tuple_count: u64,
}

/// Decode one column's terms (in id order `0..term_count`) using `decode`.
fn load_column<T, F>(data: &[u8], dir: &DictDir, decode: F) -> Result<Vec<T>>
where
    F: Fn(&[u8]) -> Result<T>,
{
    let n = dir.term_count as usize;
    // Validate the pool region and the (n+1) offset array up front — *before* any
    // allocation sized from `n`. A corrupt `term_count` (e.g. u64::MAX) must not be
    // able to drive a `Vec::with_capacity` capacity-overflow panic or an OOM abort
    // that would unwind past the caller's Err->fallback arm; bounding it by the
    // space its offset array must occupy in the file caps it at data.len()/8.
    let pool_start = dir.pool_off as usize;
    let pool_end = pool_start
        .checked_add(dir.pool_len as usize)
        .ok_or_else(|| snapshot_err("pool length overflow"))?;
    let pool = data
        .get(pool_start..pool_end)
        .ok_or_else(|| snapshot_err("string pool out of bounds"))?;

    let offsets_bytes = n
        .checked_add(1)
        .and_then(|entries| entries.checked_mul(8))
        .ok_or_else(|| snapshot_err("offset table size overflow"))?;
    let offsets_end = (dir.offsets_off as usize)
        .checked_add(offsets_bytes)
        .ok_or_else(|| snapshot_err("offset table end overflow"))?;
    if offsets_end > data.len() {
        return Err(snapshot_err("offset table out of bounds"));
    }

    let mut terms = Vec::with_capacity(n);
    let mut prev = read_u64(data, dir.offsets_off)?;
    if prev != 0 {
        return Err(snapshot_err("first pool offset must be 0"));
    }
    for i in 0..n {
        let next = read_u64(data, dir.offsets_off + ((i as u64) + 1) * 8)?;
        if next < prev || next as usize > pool.len() {
            return Err(snapshot_err("non-monotonic or out-of-range pool offset"));
        }
        let entry = &pool[prev as usize..next as usize];
        terms.push(decode(entry)?);
        prev = next;
    }
    Ok(terms)
}

/// Read one permutation index array into a `Vec<[u32;4]>`, validating the whole
/// region once and then slicing without per-element bounds checks.
fn load_index(data: &[u8], dir: &IndexDir) -> Result<Vec<[u32; 4]>> {
    let count = dir.tuple_count as usize;
    let start = dir.array_off as usize;
    let byte_len = count
        .checked_mul(16)
        .ok_or_else(|| snapshot_err("index size overflow"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| snapshot_err("index end overflow"))?;
    let region = data
        .get(start..end)
        .ok_or_else(|| snapshot_err("index array out of bounds"))?;
    let mut out = Vec::with_capacity(count);
    for chunk in region.chunks_exact(16) {
        out.push([
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
            u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
        ]);
    }
    Ok(out)
}

/// Load a `MemoryStorage` from a frozen snapshot at `path`, mmap-ing the file.
///
/// `expected_source_len`, when `Some`, is the current `data.nq` length; a
/// snapshot whose recorded `source_len` disagrees is rejected as stale so the
/// caller falls back to re-parsing the (newer) data file. Every structural
/// inconsistency, an unknown version, or a wrong-endian marker is returned as an
/// `Err` — the loader never panics on a malformed file.
pub fn load_snapshot(path: &Path, expected_source_len: Option<u64>) -> Result<MemoryStorage> {
    let file = File::open(path)
        .map_err(|e| OxirsError::Io(format!("Failed to open {}: {e}", path.display())))?;
    // SAFETY: read-only mmap of a regular file we just opened; the mapping lives
    // only for the duration of this function and is not aliased mutably. A file
    // truncated concurrently could fault, but this is the same trust model the
    // rest of oxirs-core's mmap paths use for local dataset files.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| OxirsError::Io(format!("Failed to mmap {}: {e}", path.display())))?;
    let data: &[u8] = &mmap;

    if (data.len() as u64) < DATA_START {
        return Err(snapshot_err("file smaller than header+directory"));
    }
    if &data[0..8] != MAGIC {
        return Err(snapshot_err("bad magic"));
    }
    let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    if version != FORMAT_VERSION {
        return Err(snapshot_err(format!(
            "version mismatch (file {version}, supported {FORMAT_VERSION})"
        )));
    }
    if data[12] != ENDIAN_LE {
        return Err(snapshot_err("non-little-endian snapshot"));
    }
    let quad_count = read_u64(data, 16)?;
    let source_len = read_u64(data, 24)?;
    if let Some(expected) = expected_source_len {
        if source_len != 0 && source_len != expected {
            return Err(snapshot_err(format!(
                "stale: source_len {source_len} != current data.nq {expected}"
            )));
        }
    }

    // Parse the directory.
    let mut dir_off = HEADER_LEN;
    let mut dict_dirs: Vec<DictDir> = Vec::with_capacity(4);
    for _ in 0..4 {
        let d = DictDir {
            pool_off: read_u64(data, dir_off)?,
            pool_len: read_u64(data, dir_off + 8)?,
            offsets_off: read_u64(data, dir_off + 16)?,
            term_count: read_u64(data, dir_off + 24)?,
        };
        dir_off += 32;
        dict_dirs.push(d);
    }
    let mut index_dirs: Vec<IndexDir> = Vec::with_capacity(4);
    for _ in 0..4 {
        let d = IndexDir {
            array_off: read_u64(data, dir_off)?,
            tuple_count: read_u64(data, dir_off + 8)?,
        };
        dir_off += 16;
        index_dirs.push(d);
    }
    for d in &index_dirs {
        if d.tuple_count != quad_count {
            return Err(snapshot_err(
                "permutation tuple count disagrees with header",
            ));
        }
    }

    // Decode the four columns.
    let subjects = load_column(data, &dict_dirs[0], decode_subject)?;
    let predicates = load_column(data, &dict_dirs[1], decode_predicate)?;
    let objects = load_column(data, &dict_dirs[2], decode_object)?;
    let graphs = load_column(data, &dict_dirs[3], decode_graph)?;
    let counts = [
        subjects.len(),
        predicates.len(),
        objects.len(),
        graphs.len(),
    ];

    // Read the SPOG array and derive every column's reference counts from it in a
    // single pass, bounds-checking each id against its column's term count.
    let spog_vec = load_index(data, &index_dirs[0])?;
    let mut refcounts: [Vec<u32>; 4] = [
        vec![0u32; counts[0]],
        vec![0u32; counts[1]],
        vec![0u32; counts[2]],
        vec![0u32; counts[3]],
    ];
    for tuple in &spog_vec {
        for col in 0..4 {
            let id = tuple[col] as usize;
            let slot = refcounts[col]
                .get_mut(id)
                .ok_or_else(|| snapshot_err("SPOG id out of range for its column"))?;
            *slot += 1;
        }
    }
    let [subj_rc, pred_rc, obj_rc, graph_rc] = refcounts;

    let subjects = ColumnDictionary::from_live_terms(subjects, subj_rc);
    let predicates = ColumnDictionary::from_live_terms(predicates, pred_rc);
    let objects = ColumnDictionary::from_live_terms(objects, obj_rc);
    let graphs = ColumnDictionary::from_live_terms(graphs, graph_rc);

    // Build the four permutation sets from their already-sorted arrays.
    let spog: BTreeSet<[u32; 4]> = spog_vec.into_iter().collect();
    let posg: BTreeSet<[u32; 4]> = load_index(data, &index_dirs[1])?.into_iter().collect();
    let ospg: BTreeSet<[u32; 4]> = load_index(data, &index_dirs[2])?.into_iter().collect();
    let gspo: BTreeSet<[u32; 4]> = load_index(data, &index_dirs[3])?.into_iter().collect();

    let storage = MemoryStorage::from_snapshot_parts(
        subjects, predicates, objects, graphs, spog, posg, ospg, gspo,
    );

    // The mmap was only a fast decode source; nothing borrows it now, so release
    // it here (steady-state RSS is the owned structures, not the mapped file).
    drop(mmap);

    if storage.len() as u64 != quad_count {
        return Err(snapshot_err(
            "reconstructed quad count disagrees with header",
        ));
    }
    Ok(storage)
}
