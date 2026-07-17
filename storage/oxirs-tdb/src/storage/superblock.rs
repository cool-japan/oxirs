//! On-disk superblock (page 0) for durable catalog persistence.
//!
//! The superblock is stored at page 0 of the data file and records the root
//! [`PageId`]s of every persistent B+Tree (the SPO/POS/OSP triple indexes, the
//! GSPO/GPOS/GOSP quad indexes for named graphs, plus the dictionary's
//! `term_to_id` / `id_to_term` trees), the next dictionary [`NodeId`] to hand
//! out, the total triple/quad counts, and the head of the on-disk free-page
//! list.
//!
//! Without this catalog a reopened store would rebuild empty B+Trees (all roots
//! `None`) and see zero triples even though the data pages are present on disk.
//! [`Superblock::write`] persists (and fsyncs) the catalog on `sync()`; [`Superblock::read`]
//! restores it on `open()`, after which the B+Trees are reconstructed with
//! `BTree::from_root`.
//!
//! # Format versioning
//!
//! The on-disk layout is versioned via [`SUPERBLOCK_FORMAT_VERSION`]. The first
//! two serialized fields (`magic`, `format_version`) are stable across every
//! version, so [`Superblock::read`] can decode a small [`SuperblockHeader`]
//! first and reject an incompatible file with a clear version-mismatch error —
//! rather than a confusing deserialization failure — even when newer versions
//! append trailing fields (as version 2 did for the quad-index roots).

use crate::error::{Result, TdbError};
use crate::storage::file_manager::FileManager;
use crate::storage::page::{Page, PageId, PageType};
use serde::{Deserialize, Serialize};

/// Magic number identifying an OxiRS TDB superblock ("OXIRSTDB" as big-endian ASCII).
pub const SUPERBLOCK_MAGIC: u64 = 0x4F58_4952_5354_4442;

/// Current on-disk superblock format version. Bump when the layout changes.
///
/// - Version 1: SPO/POS/OSP triple-index roots, dictionary roots, next id,
///   triple count, free-list head.
/// - Version 2: additionally persists the GSPO/GPOS/GOSP quad-index roots and
///   the named-graph quad count (F4). Version-1 files are rejected on open with
///   a clear migration error (see [`Superblock::read`]).
pub const SUPERBLOCK_FORMAT_VERSION: u32 = 2;

/// The reserved page id of the superblock itself (always page 0).
pub const SUPERBLOCK_PAGE_ID: PageId = 0;

/// Sentinel used on disk to mean "no page" (`None`).
///
/// Page 0 is permanently reserved for the superblock, so a value of `0` can
/// never be a legitimate B+Tree root or free-list entry and is therefore a safe
/// "none" sentinel.
const NONE_PAGE: u64 = 0;

/// Byte offset inside the superblock page where the length prefix begins.
const LEN_OFFSET: usize = 0;
/// Byte offset inside the superblock page where the serialized payload begins.
const PAYLOAD_OFFSET: usize = 4;

/// On-disk catalog describing where every persistent structure lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Superblock {
    /// Magic number (`SUPERBLOCK_MAGIC`) used to recognise a valid superblock.
    pub magic: u64,
    /// On-disk format version (`SUPERBLOCK_FORMAT_VERSION`).
    pub format_version: u32,
    /// Root page of the SPO triple index (`0` = empty tree).
    pub spo_root: u64,
    /// Root page of the POS triple index (`0` = empty tree).
    pub pos_root: u64,
    /// Root page of the OSP triple index (`0` = empty tree).
    pub osp_root: u64,
    /// Root page of the dictionary `term_to_id` B+Tree (`0` = empty tree).
    pub node_to_id_root: u64,
    /// Root page of the dictionary `id_to_term` B+Tree (`0` = empty tree).
    pub id_to_term_root: u64,
    /// Next dictionary [`NodeId`](crate::dictionary::NodeId) value to allocate.
    pub next_node_id: u64,
    /// Total number of triples in the default graph currently stored.
    pub triple_count: u64,
    /// Head of the persisted free-page list (`0` = empty).
    pub free_list_head: u64,
    /// Root page of the GSPO quad index (`0` = empty tree). Added in version 2.
    pub gspo_root: u64,
    /// Root page of the GPOS quad index (`0` = empty tree). Added in version 2.
    pub gpos_root: u64,
    /// Root page of the GOSP quad index (`0` = empty tree). Added in version 2.
    pub gosp_root: u64,
    /// Total number of named-graph quads currently stored. Added in version 2.
    pub quad_count: u64,
}

/// Leading, version-stable header of a [`Superblock`].
///
/// `magic` and `format_version` are always the first two serialized fields, in
/// this order, in every format version. Decoding just this header lets
/// [`Superblock::read`] validate the magic and reject an incompatible format
/// version before attempting to decode the full (possibly longer) struct.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct SuperblockHeader {
    /// Magic number (`SUPERBLOCK_MAGIC`).
    magic: u64,
    /// On-disk format version.
    format_version: u32,
}

impl SuperblockHeader {
    /// Decode only the leading `{magic, format_version}` header from a
    /// serialized superblock payload, ignoring any trailing fields.
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        oxicode::serde::decode_from_slice(bytes, oxicode::config::standard())
            .map(|(header, _)| header)
            .map_err(|e| TdbError::Deserialization(e.to_string()))
    }
}

impl Superblock {
    /// Create the superblock for a brand-new (empty) store.
    ///
    /// `next_node_id` is seeded with the first valid dictionary id so that a
    /// freshly opened store hands out ids identically whether or not it was
    /// ever reopened.
    pub fn new_empty(first_node_id: u64) -> Self {
        Superblock {
            magic: SUPERBLOCK_MAGIC,
            format_version: SUPERBLOCK_FORMAT_VERSION,
            spo_root: NONE_PAGE,
            pos_root: NONE_PAGE,
            osp_root: NONE_PAGE,
            node_to_id_root: NONE_PAGE,
            id_to_term_root: NONE_PAGE,
            next_node_id: first_node_id,
            triple_count: 0,
            free_list_head: NONE_PAGE,
            gspo_root: NONE_PAGE,
            gpos_root: NONE_PAGE,
            gosp_root: NONE_PAGE,
            quad_count: 0,
        }
    }

    /// Convert an on-disk `u64` page slot into an `Option<PageId>` (`0` = `None`).
    #[inline]
    pub fn slot_to_option(slot: u64) -> Option<PageId> {
        if slot == NONE_PAGE {
            None
        } else {
            Some(slot)
        }
    }

    /// Convert an `Option<PageId>` into its on-disk `u64` slot (`None` = `0`).
    #[inline]
    pub fn option_to_slot(page: Option<PageId>) -> u64 {
        page.unwrap_or(NONE_PAGE)
    }

    /// Root of the SPO index, or `None` for an empty tree.
    pub fn spo_root(&self) -> Option<PageId> {
        Self::slot_to_option(self.spo_root)
    }

    /// Root of the POS index, or `None` for an empty tree.
    pub fn pos_root(&self) -> Option<PageId> {
        Self::slot_to_option(self.pos_root)
    }

    /// Root of the OSP index, or `None` for an empty tree.
    pub fn osp_root(&self) -> Option<PageId> {
        Self::slot_to_option(self.osp_root)
    }

    /// Root of the dictionary `term_to_id` tree, or `None` for an empty tree.
    pub fn node_to_id_root(&self) -> Option<PageId> {
        Self::slot_to_option(self.node_to_id_root)
    }

    /// Root of the dictionary `id_to_term` tree, or `None` for an empty tree.
    pub fn id_to_term_root(&self) -> Option<PageId> {
        Self::slot_to_option(self.id_to_term_root)
    }

    /// Head of the free-page list, or `None` when there are no free pages.
    pub fn free_list_head(&self) -> Option<PageId> {
        Self::slot_to_option(self.free_list_head)
    }

    /// Root of the GSPO quad index, or `None` for an empty tree.
    pub fn gspo_root(&self) -> Option<PageId> {
        Self::slot_to_option(self.gspo_root)
    }

    /// Root of the GPOS quad index, or `None` for an empty tree.
    pub fn gpos_root(&self) -> Option<PageId> {
        Self::slot_to_option(self.gpos_root)
    }

    /// Root of the GOSP quad index, or `None` for an empty tree.
    pub fn gosp_root(&self) -> Option<PageId> {
        Self::slot_to_option(self.gosp_root)
    }

    /// Serialize the superblock into its on-disk byte representation.
    fn to_bytes(&self) -> Result<Vec<u8>> {
        oxicode::serde::encode_to_vec(self, oxicode::config::standard())
            .map_err(|e| TdbError::Serialization(e.to_string()))
    }

    /// Deserialize a superblock from its on-disk byte representation.
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        oxicode::serde::decode_from_slice(bytes, oxicode::config::standard())
            .map(|(sb, _)| sb)
            .map_err(|e| TdbError::Deserialization(e.to_string()))
    }

    /// Persist the superblock to page 0 of `file_manager`, fsyncing on the way out.
    ///
    /// The caller must ensure page 0 has already been allocated (it is reserved
    /// at store-creation time). All referenced B+Tree pages should be flushed
    /// to disk *before* the superblock is written so it never points at pages
    /// that have not yet been persisted.
    pub fn write(&self, file_manager: &FileManager) -> Result<()> {
        let bytes = self.to_bytes()?;
        let len = bytes.len();
        let len_u32 = u32::try_from(len).map_err(|_| {
            TdbError::Serialization(format!("Superblock too large to persist: {len} bytes"))
        })?;

        let mut page = Page::new(SUPERBLOCK_PAGE_ID, PageType::Metadata);
        page.write_at(LEN_OFFSET, &len_u32.to_le_bytes())?;
        page.write_at(PAYLOAD_OFFSET, &bytes)?;
        page.update_header();

        file_manager.write_page(&mut page)?;
        // Belt-and-suspenders: make sure the catalog is durable even if the
        // write_page path is later changed to defer fsync.
        file_manager.flush()?;
        Ok(())
    }

    /// Read the superblock from page 0 of `file_manager`.
    ///
    /// Returns `Ok(None)` for a brand-new, empty file (no pages yet), which the
    /// caller treats as "fresh store". Returns a clear error if page 0 exists
    /// but does not hold a recognisable, version-compatible superblock.
    pub fn read(file_manager: &FileManager) -> Result<Option<Self>> {
        if file_manager.num_pages() == 0 {
            // Brand-new file: no superblock has been written yet.
            return Ok(None);
        }

        let page = file_manager.read_page(SUPERBLOCK_PAGE_ID)?;

        let len_bytes = page.read_at(LEN_OFFSET, 4)?;
        let mut len_arr = [0u8; 4];
        len_arr.copy_from_slice(len_bytes);
        let len = u32::from_le_bytes(len_arr) as usize;

        if len == 0 || len > crate::storage::page::PAGE_USABLE_SIZE - PAYLOAD_OFFSET {
            return Err(TdbError::Other(format!(
                "Corrupt superblock at page 0: implausible payload length {len}. \
                 The data file was not created by a compatible oxirs-tdb version."
            )));
        }

        let payload = page.read_at(PAYLOAD_OFFSET, len)?;

        // Decode only the version-stable header first so that an older on-disk
        // layout (which lacks the version-2 quad fields) is rejected with a
        // clear version-mismatch error instead of a raw "not enough bytes"
        // deserialization failure from the full-struct decode below.
        let header = SuperblockHeader::from_bytes(payload)?;

        if header.magic != SUPERBLOCK_MAGIC {
            return Err(TdbError::Other(format!(
                "Not an oxirs-tdb store: superblock magic {:#018x} != {:#018x} (page 0 is not a \
                 valid superblock; the store may pre-date on-disk persistence or be corrupt)",
                header.magic, SUPERBLOCK_MAGIC
            )));
        }

        if header.format_version != SUPERBLOCK_FORMAT_VERSION {
            return Err(TdbError::Other(format!(
                "Unsupported oxirs-tdb on-disk format version {} (this build writes format \
                 version {}). The store was created by a different oxirs-tdb release; migrate \
                 or recreate it.",
                header.format_version, SUPERBLOCK_FORMAT_VERSION
            )));
        }

        let sb = Self::from_bytes(payload)?;
        Ok(Some(sb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::NodeId;
    use std::env;

    fn temp_file_manager() -> (std::path::PathBuf, FileManager) {
        let dir = env::temp_dir().join(format!("oxirs_tdb_superblock_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("data.tdb");
        let fm = FileManager::open(&path, false).expect("open file manager");
        (dir, fm)
    }

    #[test]
    fn test_superblock_read_fresh_file_is_none() {
        let (dir, fm) = temp_file_manager();
        assert!(Superblock::read(&fm).expect("read").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_superblock_round_trip() {
        let (dir, fm) = temp_file_manager();
        // Reserve page 0.
        let page0 = fm.allocate_page().expect("allocate page 0");
        assert_eq!(page0, SUPERBLOCK_PAGE_ID);

        let mut sb = Superblock::new_empty(NodeId::FIRST.as_u64());
        sb.spo_root = 7;
        sb.pos_root = 8;
        sb.osp_root = 9;
        sb.node_to_id_root = 10;
        sb.id_to_term_root = 11;
        sb.next_node_id = 42;
        sb.triple_count = 123;
        sb.free_list_head = 5;
        sb.gspo_root = 12;
        sb.gpos_root = 13;
        sb.gosp_root = 14;
        sb.quad_count = 456;

        sb.write(&fm).expect("write superblock");
        let read_back = Superblock::read(&fm)
            .expect("read")
            .expect("some superblock");
        assert_eq!(sb, read_back);
        assert_eq!(read_back.spo_root(), Some(7));
        assert_eq!(read_back.free_list_head(), Some(5));
        assert_eq!(read_back.gspo_root(), Some(12));
        assert_eq!(read_back.gpos_root(), Some(13));
        assert_eq!(read_back.gosp_root(), Some(14));
        assert_eq!(read_back.quad_count, 456);
        assert_eq!(read_back.format_version, SUPERBLOCK_FORMAT_VERSION);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_superblock_rejects_old_format_version() {
        let (dir, fm) = temp_file_manager();
        fm.allocate_page().expect("allocate page 0");

        // Write a superblock stamped with an older format version, simulating a
        // store created by a previous oxirs-tdb release. Reopening must fail
        // with a clear version-mismatch error rather than silently mis-reading.
        let mut sb = Superblock::new_empty(NodeId::FIRST.as_u64());
        sb.format_version = SUPERBLOCK_FORMAT_VERSION - 1;
        sb.write(&fm).expect("write old-format superblock");

        let err = Superblock::read(&fm).expect_err("old format must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("format version"),
            "expected a version-mismatch error, got: {msg}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_superblock_slot_option_conversions() {
        assert_eq!(Superblock::slot_to_option(0), None);
        assert_eq!(Superblock::slot_to_option(3), Some(3));
        assert_eq!(Superblock::option_to_slot(None), 0);
        assert_eq!(Superblock::option_to_slot(Some(3)), 3);
    }

    #[test]
    fn test_superblock_rejects_bad_magic() {
        let (dir, fm) = temp_file_manager();
        fm.allocate_page().expect("allocate page 0");
        // Write a page 0 that is NOT a superblock (all zeros -> len 0 / bad magic).
        let mut page = Page::new(SUPERBLOCK_PAGE_ID, PageType::Metadata);
        page.write_at(0, &7u32.to_le_bytes()).expect("write len");
        page.write_at(4, &[1u8, 2, 3, 4, 5, 6, 7])
            .expect("write junk");
        page.update_header();
        fm.write_page(&mut page).expect("write junk page");

        assert!(Superblock::read(&fm).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
