//! Integration tests for the Modbus TUI register browser.
//!
//! These tests exercise state and logic only — no terminal is launched.
//! All tests run without a real Modbus device or display.

#![cfg(feature = "tui")]

use oxirs_modbus::browser::{
    state::{BrowserState, FilteredView, RegisterEntry, RegisterTab, RegisterValue},
    BrowserConfig, RegisterBrowser,
};

// ──────────────────────────────────────────────────────────────────────────────
// RegisterTab tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn tab_all_returns_four_variants() {
    let all = RegisterTab::all();
    assert_eq!(all.len(), 4);
    assert!(all.contains(&RegisterTab::Coils));
    assert!(all.contains(&RegisterTab::DiscreteInputs));
    assert!(all.contains(&RegisterTab::HoldingRegisters));
    assert!(all.contains(&RegisterTab::InputRegisters));
}

#[test]
fn tab_next_cycles_forward() {
    assert_eq!(RegisterTab::Coils.next(), RegisterTab::DiscreteInputs);
    assert_eq!(
        RegisterTab::DiscreteInputs.next(),
        RegisterTab::HoldingRegisters
    );
    assert_eq!(
        RegisterTab::HoldingRegisters.next(),
        RegisterTab::InputRegisters
    );
    assert_eq!(RegisterTab::InputRegisters.next(), RegisterTab::Coils);
}

#[test]
fn tab_prev_cycles_backward() {
    assert_eq!(RegisterTab::Coils.prev(), RegisterTab::InputRegisters);
    assert_eq!(
        RegisterTab::InputRegisters.prev(),
        RegisterTab::HoldingRegisters
    );
    assert_eq!(
        RegisterTab::HoldingRegisters.prev(),
        RegisterTab::DiscreteInputs
    );
    assert_eq!(RegisterTab::DiscreteInputs.prev(), RegisterTab::Coils);
}

#[test]
fn tab_titles_are_nonempty() {
    for tab in RegisterTab::all() {
        assert!(!tab.title().is_empty(), "tab {:?} has empty title", tab);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RegisterValue tests
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn register_value_display_bool_on() {
    assert_eq!(RegisterValue::Bool(true).display(), "ON");
}

#[test]
fn register_value_display_bool_off() {
    assert_eq!(RegisterValue::Bool(false).display(), "OFF");
}

#[test]
fn register_value_display_u16() {
    assert_eq!(RegisterValue::U16(42).display(), "42");
    assert_eq!(RegisterValue::U16(0).display(), "0");
    assert_eq!(RegisterValue::U16(65535).display(), "65535");
}

#[test]
fn register_value_display_f32() {
    // Use a value that is not a well-known constant (avoid clippy::approx_constant).
    let s = RegisterValue::F32(23.75_f32).display();
    assert!(s.starts_with("23.75"), "got: {s}");
}

#[test]
fn register_value_display_unknown() {
    assert_eq!(RegisterValue::Unknown.display(), "—");
}

// ──────────────────────────────────────────────────────────────────────────────
// BrowserState initialisation
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn browser_state_new_defaults() {
    let state = BrowserState::new(500);
    assert_eq!(state.active_tab, RegisterTab::Coils);
    assert_eq!(state.selected_row, 0);
    assert_eq!(state.scroll_offset, 0);
    assert!(state.filter.is_empty());
    assert!(!state.edit_mode);
    assert!(!state.show_help);
    assert_eq!(state.poll_interval_ms, 500);
    // All four tabs should have empty register lists.
    for tab in RegisterTab::all() {
        assert!(
            state.registers.get(&tab).map(Vec::is_empty).unwrap_or(true),
            "tab {:?} should start empty",
            tab
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scroll / selection tests
// ──────────────────────────────────────────────────────────────────────────────

fn state_with_registers(count: u16) -> BrowserState {
    let mut state = BrowserState::new(500);
    for i in 0..count {
        state.update_register(RegisterTab::HoldingRegisters, i, RegisterValue::U16(i * 10));
    }
    state.next_tab(); // DiscreteInputs
    state.next_tab(); // HoldingRegisters
    state
}

#[test]
fn scroll_down_increments_selected_row() {
    let mut state = state_with_registers(5);
    assert_eq!(state.selected_row, 0);
    state.scroll_down();
    assert_eq!(state.selected_row, 1);
    state.scroll_down();
    assert_eq!(state.selected_row, 2);
}

#[test]
fn scroll_down_clamps_at_last_entry() {
    let mut state = state_with_registers(3);
    state.scroll_down();
    state.scroll_down();
    state.scroll_down(); // would go to 3, but max is 2
    state.scroll_down();
    assert_eq!(state.selected_row, 2, "should clamp at index 2 (3 entries)");
}

#[test]
fn scroll_up_clamps_at_zero() {
    let mut state = state_with_registers(3);
    state.scroll_up(); // already at 0
    assert_eq!(state.selected_row, 0);
}

#[test]
fn scroll_up_after_scroll_down() {
    let mut state = state_with_registers(5);
    state.scroll_down();
    state.scroll_down();
    assert_eq!(state.selected_row, 2);
    state.scroll_up();
    assert_eq!(state.selected_row, 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// Tab navigation
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn next_tab_advances_and_resets_selection() {
    let mut state = state_with_registers(5);
    state.scroll_down();
    state.scroll_down();
    assert_eq!(state.selected_row, 2);
    state.next_tab(); // wraps from HoldingRegisters → InputRegisters
    assert_eq!(
        state.selected_row, 0,
        "selection should reset on tab change"
    );
}

#[test]
fn prev_tab_retreats_and_resets_selection() {
    let mut state = state_with_registers(5);
    state.scroll_down();
    state.prev_tab(); // HoldingRegisters → DiscreteInputs
    assert_eq!(state.selected_row, 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// update_register
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn update_register_inserts_new_entry() {
    let mut state = BrowserState::new(500);
    state.update_register(RegisterTab::Coils, 10, RegisterValue::Bool(true));
    let regs = state.registers.get(&RegisterTab::Coils).expect("coils");
    assert_eq!(regs.len(), 1);
    assert_eq!(regs[0].address, 10);
    assert!(matches!(regs[0].value, RegisterValue::Bool(true)));
}

#[test]
fn update_register_replaces_existing_entry() {
    let mut state = BrowserState::new(500);
    state.update_register(RegisterTab::HoldingRegisters, 5, RegisterValue::U16(100));
    state.update_register(RegisterTab::HoldingRegisters, 5, RegisterValue::U16(200));
    let regs = state
        .registers
        .get(&RegisterTab::HoldingRegisters)
        .expect("holding");
    assert_eq!(
        regs.len(),
        1,
        "should not duplicate entries for same address"
    );
    assert!(matches!(regs[0].value, RegisterValue::U16(200)));
}

#[test]
fn update_register_sorts_by_address() {
    let mut state = BrowserState::new(500);
    state.update_register(RegisterTab::HoldingRegisters, 30, RegisterValue::U16(1));
    state.update_register(RegisterTab::HoldingRegisters, 10, RegisterValue::U16(2));
    state.update_register(RegisterTab::HoldingRegisters, 20, RegisterValue::U16(3));
    let regs = state
        .registers
        .get(&RegisterTab::HoldingRegisters)
        .expect("holding");
    let addresses: Vec<u16> = regs.iter().map(|e| e.address).collect();
    assert_eq!(addresses, vec![10, 20, 30]);
}

// ──────────────────────────────────────────────────────────────────────────────
// filtered_registers
// ──────────────────────────────────────────────────────────────────────────────

fn state_with_labeled_registers() -> BrowserState {
    let mut state = BrowserState::new(500);
    state.update_register(RegisterTab::HoldingRegisters, 0, RegisterValue::U16(1));
    state.update_register(RegisterTab::HoldingRegisters, 1, RegisterValue::U16(2));
    state.update_register(RegisterTab::HoldingRegisters, 2, RegisterValue::U16(3));
    // Add labels manually.
    if let Some(regs) = state.registers.get_mut(&RegisterTab::HoldingRegisters) {
        regs[0].label = Some("Temperature".to_string());
        regs[1].label = Some("Pressure".to_string());
        regs[2].label = Some("Flow rate".to_string());
    }
    // Switch to HoldingRegisters tab.
    state.next_tab(); // DiscreteInputs
    state.next_tab(); // HoldingRegisters
    state
}

#[test]
fn filtered_registers_empty_filter_returns_all() {
    let state = state_with_labeled_registers();
    assert!(state.filter.is_empty());
    assert_eq!(state.filtered_registers().len(), 3);
}

#[test]
fn filtered_registers_by_address_decimal() {
    let mut state = state_with_labeled_registers();
    state.filter = "1".to_string();
    let filtered = state.filtered_registers();
    // Address "1" matches address 1.
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].address, 1);
}

#[test]
fn filtered_registers_by_label() {
    let mut state = state_with_labeled_registers();
    state.filter = "temp".to_string();
    let filtered = state.filtered_registers();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].address, 0);
}

#[test]
fn filtered_registers_no_match_returns_empty() {
    let mut state = state_with_labeled_registers();
    state.filter = "zzz_no_match".to_string();
    assert_eq!(state.filtered_registers().len(), 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// FilteredView
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn filtered_view_total_equals_filtered_len() {
    let mut state = state_with_labeled_registers();
    state.filter = "pressure".to_string();
    let view: FilteredView<'_> = state.filtered_view();
    assert_eq!(view.total, view.entries.len());
    assert_eq!(view.total, 1);
}

#[test]
fn filtered_view_total_is_after_filter_not_raw() {
    let mut state = state_with_labeled_registers();
    // All 3 raw entries, filter returns 1.
    state.filter = "flow".to_string();
    let view = state.filtered_view();
    assert_eq!(
        view.total, 1,
        "total should reflect filtered count, not raw"
    );
    assert_eq!(view.entries.len(), 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// BrowserConfig defaults
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn browser_config_default_has_reasonable_values() {
    let cfg = BrowserConfig::default();
    assert!(cfg.poll_interval_ms > 0);
    assert!(cfg.max_address > 0);
    assert!(cfg.device_id > 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// RegisterBrowser construction and load_snapshot
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn register_browser_new_and_state_accessible() {
    let browser = RegisterBrowser::new(BrowserConfig::default());
    let state = browser.state();
    assert_eq!(state.active_tab, RegisterTab::Coils);
}

#[test]
fn load_snapshot_populates_state() {
    let mut browser = RegisterBrowser::new(BrowserConfig::default());
    browser.load_snapshot(
        RegisterTab::HoldingRegisters,
        vec![
            (10, RegisterValue::U16(100), Some("Motor speed".to_string())),
            (11, RegisterValue::F32(23.5), None),
        ],
    );
    let state = browser.state();
    let regs = state
        .registers
        .get(&RegisterTab::HoldingRegisters)
        .expect("holding registers");
    assert_eq!(regs.len(), 2);
    assert_eq!(regs[0].address, 10);
    assert_eq!(regs[0].label.as_deref(), Some("Motor speed"));
    assert_eq!(regs[1].address, 11);
    assert!(regs[1].label.is_none());
}

#[test]
fn load_snapshot_multiple_tabs() {
    let mut browser = RegisterBrowser::new(BrowserConfig::default());
    browser.load_snapshot(
        RegisterTab::Coils,
        vec![(0, RegisterValue::Bool(true), Some("Pump 1".to_string()))],
    );
    browser.load_snapshot(
        RegisterTab::DiscreteInputs,
        vec![(5, RegisterValue::Bool(false), None)],
    );
    let state = browser.state();
    assert_eq!(
        state.registers.get(&RegisterTab::Coils).map(Vec::len),
        Some(1)
    );
    assert_eq!(
        state
            .registers
            .get(&RegisterTab::DiscreteInputs)
            .map(Vec::len),
        Some(1)
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// set_error
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn set_error_records_error_on_existing_entry() {
    let mut state = BrowserState::new(500);
    state.update_register(RegisterTab::InputRegisters, 0, RegisterValue::U16(42));
    state.set_error(RegisterTab::InputRegisters, 0, "timeout".to_string());
    let regs = state
        .registers
        .get(&RegisterTab::InputRegisters)
        .expect("input");
    assert_eq!(regs[0].error.as_deref(), Some("timeout"));
}

#[test]
fn set_error_creates_entry_if_missing() {
    let mut state = BrowserState::new(500);
    state.set_error(
        RegisterTab::InputRegisters,
        99,
        "device offline".to_string(),
    );
    let regs = state
        .registers
        .get(&RegisterTab::InputRegisters)
        .expect("input");
    assert_eq!(regs.len(), 1);
    assert_eq!(regs[0].address, 99);
    assert!(matches!(regs[0].value, RegisterValue::Unknown));
}

// ──────────────────────────────────────────────────────────────────────────────
// widgets helpers
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn format_hex_pads_to_four_digits() {
    use oxirs_modbus::browser::widgets::format_hex;
    assert_eq!(format_hex(0x0000), "0x0000");
    assert_eq!(format_hex(0x0042), "0x0042");
    assert_eq!(format_hex(0xFFFF), "0xFFFF");
}

#[test]
fn format_binary_groups_nibbles() {
    use oxirs_modbus::browser::widgets::format_binary;
    assert_eq!(format_binary(0x0042), "0000_0000_0100_0010");
    assert_eq!(format_binary(0x0000), "0000_0000_0000_0000");
    assert_eq!(format_binary(0xFFFF), "1111_1111_1111_1111");
}
