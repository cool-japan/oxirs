//! [`RegisterBrowser`]: interactive TUI application for browsing Modbus registers.
//!
//! The browser takes over the terminal (raw mode + alternate screen) and
//! restores it on exit — even if an error or panic occurs — via
//! `TerminalGuard`.
//!
//! # Key bindings
//!
//! | Key | Action |
//! |-----|--------|
//! | `Tab` / `Shift+Tab` | Next / previous tab |
//! | `j` / `k`, `↓` / `↑` | Scroll down / up |
//! | `/` | Enter filter mode |
//! | `Enter` | Confirm filter; in normal mode toggle edit |
//! | `e` | Enter edit mode for selected holding register |
//! | `Esc` | Exit filter / edit mode |
//! | `r` | Force refresh (re-poll) |
//! | `?` | Toggle help overlay |
//! | `q` / `Ctrl+C` | Quit |

use std::io;
use std::time::{Duration, SystemTime};

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use super::{
    state::{BrowserState, RegisterTab, RegisterValue},
    ui,
};

// ──────────────────────────────────────────────────────────────────────────────
// BrowserConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the register browser.
#[derive(Debug, Clone)]
pub struct BrowserConfig {
    /// How often to redraw the screen (does not drive real polling).
    pub poll_interval_ms: u64,
    /// Maximum register address to display (exclusive upper bound).
    pub max_address: u16,
    /// Modbus device / unit ID (informational).
    pub device_id: u8,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 500,
            max_address: 100,
            device_id: 1,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BrowserError
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during browser operation.
#[derive(Debug, thiserror::Error)]
pub enum BrowserError {
    /// Terminal initialisation or teardown error.
    #[error("terminal error: {0}")]
    Terminal(String),
    /// Underlying I/O error.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Modbus protocol or device error (for future use when live polling is added).
    #[error("modbus error: {0}")]
    Modbus(String),
}

// ──────────────────────────────────────────────────────────────────────────────
// TerminalGuard — RAII cleanup
// ──────────────────────────────────────────────────────────────────────────────

/// RAII guard that restores the terminal on drop.
///
/// This ensures the terminal is always cleaned up even if `run()` returns
/// an error or a panic unwinds the stack.
struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalGuard {
    /// Enter raw mode + alternate screen and wrap the resulting terminal.
    fn enter() -> Result<Self, BrowserError> {
        enable_raw_mode().map_err(|e| BrowserError::Terminal(e.to_string()))?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)
            .map_err(|e| BrowserError::Terminal(e.to_string()))?;
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend).map_err(|e| BrowserError::Terminal(e.to_string()))?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        // Best-effort cleanup; ignore errors because we're in Drop.
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RegisterBrowser
// ──────────────────────────────────────────────────────────────────────────────

/// Interactive TUI register browser.
///
/// # Minimal usage
///
/// ```no_run
/// # #[cfg(feature = "tui")]
/// # {
/// use oxirs_modbus::browser::{RegisterBrowser, BrowserConfig, BrowserError};
/// use oxirs_modbus::browser::state::{RegisterTab, RegisterValue};
///
/// let mut browser = RegisterBrowser::new(BrowserConfig::default());
/// browser.load_snapshot(
///     RegisterTab::HoldingRegisters,
///     vec![
///         (0, RegisterValue::U16(1234), Some("Temperature".to_string())),
///         (1, RegisterValue::F32(23.5), Some("Setpoint".to_string())),
///     ],
/// );
/// browser.run()?;
/// # }
/// # Ok::<(), oxirs_modbus::browser::BrowserError>(())
/// ```
pub struct RegisterBrowser {
    config: BrowserConfig,
    state: BrowserState,
}

impl RegisterBrowser {
    /// Create a new browser with the given configuration.
    pub fn new(config: BrowserConfig) -> Self {
        let poll_ms = config.poll_interval_ms;
        Self {
            config,
            state: BrowserState::new(poll_ms),
        }
    }

    /// Load a snapshot of register data without connecting to a real device.
    ///
    /// Each tuple is `(address, value, label)`.  Entries are added to `tab` in
    /// address order.  Existing entries are replaced if the address matches.
    pub fn load_snapshot(
        &mut self,
        tab: RegisterTab,
        entries: Vec<(u16, RegisterValue, Option<String>)>,
    ) {
        for (address, value, label) in entries {
            self.state.update_register(tab, address, value);
            // Attach the label after the update.
            if let Some(ref lbl) = label {
                if let Some(regs) = self.state.registers.get_mut(&tab) {
                    if let Some(entry) = regs.iter_mut().find(|e| e.address == address) {
                        entry.label = Some(lbl.clone());
                    }
                }
            }
        }
    }

    /// Provide read-only access to the current browser state.
    ///
    /// Useful in tests to verify state without launching the TUI.
    pub fn state(&self) -> &BrowserState {
        &self.state
    }

    /// Run the interactive TUI.  Returns when the user presses `q` or `Ctrl+C`.
    ///
    /// This method:
    /// 1. Switches the terminal to raw mode and alternate screen.
    /// 2. Runs the event loop until the user quits.
    /// 3. Restores the terminal (via `TerminalGuard` drop) before returning.
    pub fn run(mut self) -> Result<(), BrowserError> {
        let mut guard = TerminalGuard::enter()?;
        let tick = Duration::from_millis(self.config.poll_interval_ms.max(50));

        loop {
            guard
                .terminal
                .draw(|frame| ui::render(frame, &self.state))
                .map_err(|e| BrowserError::Terminal(e.to_string()))?;

            if event::poll(tick).map_err(|e| BrowserError::Terminal(e.to_string()))? {
                if let Event::Key(key) = event::read().map_err(BrowserError::Io)? {
                    if handle_key(&mut self.state, key.code, key.modifiers) {
                        break;
                    }
                }
            }
        }

        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Key handler
// ──────────────────────────────────────────────────────────────────────────────

/// Handle a single key event.  Returns `true` if the application should quit.
fn handle_key(state: &mut BrowserState, code: KeyCode, modifiers: KeyModifiers) -> bool {
    // Ctrl+C always quits.
    if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
        return true;
    }

    // Edit mode: forward characters to the edit buffer.
    if state.edit_mode {
        return handle_edit_key(state, code);
    }

    // Filter-typing mode: forward characters into state.filter.
    if state.filter_active {
        return handle_filter_key(state, code);
    }

    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Tab => state.next_tab(),
        KeyCode::BackTab => state.prev_tab(),
        KeyCode::Char('j') | KeyCode::Down => state.scroll_down(),
        KeyCode::Char('k') | KeyCode::Up => state.scroll_up(),
        KeyCode::Char('/') => {
            state.filter_active = true;
            state.filter.clear();
        }
        KeyCode::Char('e') => {
            if matches!(
                state.active_tab,
                crate::browser::state::RegisterTab::HoldingRegisters
            ) {
                state.toggle_edit_mode();
            } else {
                state.status_message =
                    Some("Edit only available for Holding Registers".to_string());
            }
        }
        KeyCode::Char('r') => {
            // Mark all entries with a "refreshed" timestamp for visual feedback.
            let now = SystemTime::now();
            for entries in state.registers.values_mut() {
                for e in entries.iter_mut() {
                    e.last_updated = now;
                }
            }
            state.status_message = Some("Refreshed".to_string());
        }
        KeyCode::Char('?') => {
            state.show_help = !state.show_help;
        }
        KeyCode::Esc => {
            state.show_help = false;
            state.status_message = None;
        }
        KeyCode::Enter => {
            // In normal mode, Enter confirms any pending status or toggles edit.
            state.status_message = None;
        }
        _ => {}
    }
    false
}

/// Handle a key while in filter mode.  Returns `true` if the app should quit.
fn handle_filter_key(state: &mut BrowserState, code: KeyCode) -> bool {
    match code {
        KeyCode::Esc => {
            state.filter.clear();
            state.filter_active = false;
        }
        KeyCode::Enter => {
            // Confirm filter — stay in the view with the filter applied.
            state.filter_active = false;
        }
        KeyCode::Backspace => {
            state.filter.pop();
        }
        KeyCode::Char(c) => {
            state.filter.push(c);
            state.clamp_selection();
        }
        _ => {}
    }
    false
}

/// Handle a key while in edit mode.  Returns `true` if the app should quit.
fn handle_edit_key(state: &mut BrowserState, code: KeyCode) -> bool {
    match code {
        KeyCode::Esc => {
            state.edit_mode = false;
            state.edit_buffer.clear();
        }
        KeyCode::Enter => {
            // Parse the buffer and apply the value to the selected register.
            apply_edit_value(state);
            state.edit_mode = false;
            state.edit_buffer.clear();
        }
        KeyCode::Backspace => {
            state.edit_buffer.pop();
        }
        KeyCode::Char(c) => {
            state.edit_buffer.push(c);
        }
        _ => {}
    }
    false
}

/// Parse `state.edit_buffer` and write the value to the selected register.
fn apply_edit_value(state: &mut BrowserState) {
    let address = match state.selected_entry().map(|e| e.address) {
        Some(a) => a,
        None => {
            state.status_message = Some("No register selected".to_string());
            return;
        }
    };

    let raw_str = state.edit_buffer.trim().to_string();
    let value = if let Ok(v) = raw_str.parse::<u16>() {
        RegisterValue::U16(v)
    } else if raw_str.starts_with("0x") || raw_str.starts_with("0X") {
        match u16::from_str_radix(&raw_str[2..], 16) {
            Ok(v) => RegisterValue::U16(v),
            Err(_) => {
                state.status_message = Some(format!("Invalid hex: {raw_str}"));
                return;
            }
        }
    } else if let Ok(v) = raw_str.parse::<f32>() {
        RegisterValue::F32(v)
    } else {
        state.status_message = Some(format!("Cannot parse: {raw_str}"));
        return;
    };

    state.update_register(RegisterTab::HoldingRegisters, address, value);
    state.status_message = Some(format!("Updated register 0x{:04X}", address));
}
