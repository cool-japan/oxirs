//! Browser state: register entries, tabs, filtering, and editing.
//!
//! The [`BrowserState`] holds all mutable data for the TUI register browser.
//! Selected-row indices are always treated as indices into the **filtered** view
//! (the result of [`BrowserState::filtered_registers`]), not the raw unfiltered
//! `Vec<RegisterEntry>`.  All mutation methods maintain this invariant.

use std::collections::HashMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// RegisterTab
// ──────────────────────────────────────────────────────────────────────────────

/// The four Modbus register spaces, each shown on a separate tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterTab {
    /// Coils (FC01 / FC05 / FC15): 1-bit read-write.
    Coils,
    /// Discrete Inputs (FC02): 1-bit read-only.
    DiscreteInputs,
    /// Holding Registers (FC03 / FC06 / FC16): 16-bit read-write.
    HoldingRegisters,
    /// Input Registers (FC04): 16-bit read-only.
    InputRegisters,
}

impl RegisterTab {
    /// Short display title for the tab bar.
    pub fn title(self) -> &'static str {
        match self {
            Self::Coils => "Coils",
            Self::DiscreteInputs => "Discrete Inputs",
            Self::HoldingRegisters => "Holding Registers",
            Self::InputRegisters => "Input Registers",
        }
    }

    /// All four tab variants in display order.
    pub fn all() -> [RegisterTab; 4] {
        [
            Self::Coils,
            Self::DiscreteInputs,
            Self::HoldingRegisters,
            Self::InputRegisters,
        ]
    }

    /// Cycle to the next tab (wraps around).
    pub fn next(self) -> Self {
        match self {
            Self::Coils => Self::DiscreteInputs,
            Self::DiscreteInputs => Self::HoldingRegisters,
            Self::HoldingRegisters => Self::InputRegisters,
            Self::InputRegisters => Self::Coils,
        }
    }

    /// Cycle to the previous tab (wraps around).
    pub fn prev(self) -> Self {
        match self {
            Self::Coils => Self::InputRegisters,
            Self::DiscreteInputs => Self::Coils,
            Self::HoldingRegisters => Self::DiscreteInputs,
            Self::InputRegisters => Self::HoldingRegisters,
        }
    }

    /// Tab index in the `all()` array (0-based).
    pub fn index(self) -> usize {
        match self {
            Self::Coils => 0,
            Self::DiscreteInputs => 1,
            Self::HoldingRegisters => 2,
            Self::InputRegisters => 3,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RegisterValue
// ──────────────────────────────────────────────────────────────────────────────

/// The decoded value stored in a register entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegisterValue {
    /// Boolean: used for coils and discrete inputs.
    Bool(bool),
    /// Raw unsigned 16-bit word: holding/input registers (uninterpreted).
    U16(u16),
    /// IEEE 754 single-precision float spanning two consecutive registers.
    F32(f32),
    /// Value could not be determined.
    Unknown,
}

impl RegisterValue {
    /// Human-readable string for display in the TUI table.
    pub fn display(&self) -> String {
        match self {
            Self::Bool(b) => {
                if *b {
                    "ON".to_string()
                } else {
                    "OFF".to_string()
                }
            }
            Self::U16(v) => format!("{v}"),
            Self::F32(v) => format!("{v:.4}"),
            Self::Unknown => "—".to_string(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RegisterEntry
// ──────────────────────────────────────────────────────────────────────────────

/// A single register entry in the live browser view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterEntry {
    /// Modbus register address (0-based Modbus address space).
    pub address: u16,
    /// Current decoded value.
    pub value: RegisterValue,
    /// Optional human-readable label from the register map, if known.
    pub label: Option<String>,
    /// Timestamp of the last successful read.
    ///
    /// Skipped during serialisation; restored to `UNIX_EPOCH` on deserialisation.
    #[serde(skip, default = "SystemTime::now")]
    pub last_updated: SystemTime,
    /// Last read error for this register, if any.
    pub error: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// FilteredView
// ──────────────────────────────────────────────────────────────────────────────

/// A snapshot of filtered/sorted register entries ready for rendering.
///
/// `total` is the number of entries **after** filtering (not the unfiltered
/// count).  `visible_start` is the scroll offset into `entries`.
pub struct FilteredView<'a> {
    /// All entries that pass the current filter.
    pub entries: Vec<&'a RegisterEntry>,
    /// Number of entries after filtering (equals `entries.len()`).
    pub total: usize,
    /// Scroll offset: index into `entries` of the first visible row.
    pub visible_start: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// BrowserState
// ──────────────────────────────────────────────────────────────────────────────

/// All mutable state for the TUI register browser.
pub struct BrowserState {
    /// Which register tab is currently active.
    pub active_tab: RegisterTab,
    /// Register entries keyed by tab.
    pub registers: HashMap<RegisterTab, Vec<RegisterEntry>>,
    /// Number of rows scrolled past the top of the filtered view.
    pub scroll_offset: usize,
    /// Row index within the **filtered** view that is highlighted.
    pub selected_row: usize,
    /// Free-text filter applied to address (decimal/hex prefix) or label.
    pub filter: String,
    /// `true` while the user is actively typing into the filter box.
    pub filter_active: bool,
    /// Whether the user is currently editing a register value.
    pub edit_mode: bool,
    /// Input buffer while in edit mode.
    pub edit_buffer: String,
    /// One-shot status message shown in the bottom bar.
    pub status_message: Option<String>,
    /// Desired poll interval in milliseconds (informational; not driven here).
    pub poll_interval_ms: u64,
    /// Whether the help overlay is visible.
    pub show_help: bool,
}

impl BrowserState {
    /// Create a new [`BrowserState`] with empty register lists.
    pub fn new(poll_interval_ms: u64) -> Self {
        let mut registers = HashMap::new();
        for tab in RegisterTab::all() {
            registers.insert(tab, Vec::new());
        }
        Self {
            active_tab: RegisterTab::Coils,
            registers,
            scroll_offset: 0,
            selected_row: 0,
            filter: String::new(),
            filter_active: false,
            edit_mode: false,
            edit_buffer: String::new(),
            status_message: None,
            poll_interval_ms,
            show_help: false,
        }
    }

    /// All entries in the currently-active tab (unfiltered).
    pub fn current_tab_registers(&self) -> &[RegisterEntry] {
        self.registers
            .get(&self.active_tab)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Return entries from the active tab that match the current filter.
    ///
    /// The filter is matched case-insensitively against:
    /// - decimal address string (`"42"`)
    /// - zero-padded hex address (`"002A"`)
    /// - label, if present
    pub fn filtered_registers(&self) -> Vec<&RegisterEntry> {
        let entries = self.current_tab_registers();
        if self.filter.is_empty() {
            return entries.iter().collect();
        }
        let lc = self.filter.to_lowercase();
        entries
            .iter()
            .filter(|e| {
                let addr_dec = e.address.to_string();
                let addr_hex = format!("{:04X}", e.address).to_lowercase();
                let label_match = e
                    .label
                    .as_deref()
                    .map(|l| l.to_lowercase().contains(&lc))
                    .unwrap_or(false);
                addr_dec.contains(&lc) || addr_hex.contains(&lc) || label_match
            })
            .collect()
    }

    /// Build a [`FilteredView`] for the current state.
    pub fn filtered_view(&self) -> FilteredView<'_> {
        let entries = self.filtered_registers();
        let total = entries.len();
        FilteredView {
            entries,
            total,
            visible_start: self.scroll_offset,
        }
    }

    /// The currently-selected register entry, or `None` if the list is empty.
    pub fn selected_entry(&self) -> Option<&RegisterEntry> {
        let filtered = self.filtered_registers();
        filtered.get(self.selected_row).copied()
    }

    /// Move selection down by one row, clamping at the bottom of the filtered view.
    pub fn scroll_down(&mut self) {
        let max = self.filtered_registers().len().saturating_sub(1);
        if self.selected_row < max {
            self.selected_row += 1;
        }
    }

    /// Move selection up by one row, clamping at 0.
    pub fn scroll_up(&mut self) {
        if self.selected_row > 0 {
            self.selected_row -= 1;
        }
    }

    /// Advance to the next tab and reset scroll/selection.
    pub fn next_tab(&mut self) {
        self.active_tab = self.active_tab.next();
        self.reset_view();
    }

    /// Go to the previous tab and reset scroll/selection.
    pub fn prev_tab(&mut self) {
        self.active_tab = self.active_tab.prev();
        self.reset_view();
    }

    /// Toggle whether the edit buffer is active (for holding register writes).
    pub fn toggle_edit_mode(&mut self) {
        self.edit_mode = !self.edit_mode;
        if self.edit_mode {
            // Pre-fill buffer with the current raw value, if available.
            self.edit_buffer = self
                .selected_entry()
                .map(|e| e.value.display())
                .unwrap_or_default();
        } else {
            self.edit_buffer.clear();
        }
    }

    /// Insert or update a register entry in `tab`.
    ///
    /// If an entry for `address` already exists it is replaced in-place.
    pub fn update_register(&mut self, tab: RegisterTab, address: u16, value: RegisterValue) {
        let entries = self.registers.entry(tab).or_default();
        if let Some(entry) = entries.iter_mut().find(|e| e.address == address) {
            entry.value = value;
            entry.last_updated = SystemTime::now();
            entry.error = None;
        } else {
            entries.push(RegisterEntry {
                address,
                value,
                label: None,
                last_updated: SystemTime::now(),
                error: None,
            });
            entries.sort_by_key(|e| e.address);
        }
        self.clamp_selection();
    }

    /// Record a read error for a specific register address.
    pub fn set_error(&mut self, tab: RegisterTab, address: u16, error: String) {
        let entries = self.registers.entry(tab).or_default();
        if let Some(entry) = entries.iter_mut().find(|e| e.address == address) {
            entry.error = Some(error);
        } else {
            entries.push(RegisterEntry {
                address,
                value: RegisterValue::Unknown,
                label: None,
                last_updated: SystemTime::now(),
                error: Some(error),
            });
            entries.sort_by_key(|e| e.address);
        }
        self.clamp_selection();
    }

    /// Apply a new filter string, then re-clamp selection so it stays valid.
    pub fn set_filter(&mut self, filter: String) {
        self.filter = filter;
        self.reset_view();
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Reset scroll and selection to the top; call after tab/filter changes.
    fn reset_view(&mut self) {
        self.scroll_offset = 0;
        self.selected_row = 0;
    }

    /// Clamp `selected_row` to the current filtered-view length.
    ///
    /// Called after filter or register updates so the highlighted row stays valid.
    pub fn clamp_selection(&mut self) {
        let count = self.filtered_registers().len();
        if count == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= count {
            self.selected_row = count - 1;
        }
    }
}
