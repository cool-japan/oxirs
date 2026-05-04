//! Standalone widget helpers for the Modbus register browser.
//!
//! These are pure-data helpers that produce ratatui widgets or format strings.
//! They do not hold any mutable state.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

// ──────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Format a 16-bit unsigned value as a zero-padded hexadecimal string.
///
/// ```
/// # use oxirs_modbus::browser::widgets::format_hex;
/// assert_eq!(format_hex(0x0042), "0x0042");
/// assert_eq!(format_hex(0xFFFF), "0xFFFF");
/// ```
pub fn format_hex(value: u16) -> String {
    format!("0x{:04X}", value)
}

/// Format a 16-bit unsigned value as a grouped binary string (nibble groups).
///
/// ```
/// # use oxirs_modbus::browser::widgets::format_binary;
/// assert_eq!(format_binary(0x0042), "0000_0000_0100_0010");
/// ```
pub fn format_binary(value: u16) -> String {
    let bits = format!("{:016b}", value);
    // Insert underscores every 4 bits: xxxx_xxxx_xxxx_xxxx
    let groups: Vec<&str> = bits
        .as_bytes()
        .chunks(4)
        .map(|c| std::str::from_utf8(c).unwrap_or("????"))
        .collect();
    groups.join("_")
}

// ──────────────────────────────────────────────────────────────────────────────
// Help overlay
// ──────────────────────────────────────────────────────────────────────────────

/// Build the help overlay paragraph widget.
///
/// This is a static widget — callers position it with
/// [`ratatui::layout::Rect`] via [`ratatui::Frame::render_widget`].
pub fn help_overlay_widget() -> Paragraph<'static> {
    let key_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let desc_style = Style::default().fg(Color::White);

    let lines: Vec<Line<'static>> = vec![
        Line::from(vec![Span::styled(
            " Keyboard Shortcuts ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Tab / Shift+Tab  ", key_style),
            Span::styled("Next / previous tab", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  j / k  ↓ / ↑    ", key_style),
            Span::styled("Scroll register list", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  /                ", key_style),
            Span::styled("Enter filter mode", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  Enter            ", key_style),
            Span::styled("Confirm filter / toggle edit", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  e                ", key_style),
            Span::styled("Edit selected holding register", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  Esc              ", key_style),
            Span::styled("Exit filter / edit mode", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  r                ", key_style),
            Span::styled("Force refresh (re-poll)", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  ?                ", key_style),
            Span::styled("Toggle this help overlay", desc_style),
        ]),
        Line::from(vec![
            Span::styled("  q / Ctrl+C       ", key_style),
            Span::styled("Quit", desc_style),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Press ? or Esc to close",
            Style::default().fg(Color::DarkGray),
        )]),
    ];

    Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Help ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: false })
}
