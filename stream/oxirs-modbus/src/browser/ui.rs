//! TUI rendering: translates [`BrowserState`] into ratatui draw calls.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, Tabs},
    Frame,
};

use super::state::{BrowserState, RegisterTab, RegisterValue};
use super::widgets::help_overlay_widget;

// ──────────────────────────────────────────────────────────────────────────────
// Public entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Render the full register browser UI into a ratatui [`Frame`].
///
/// Layout:
/// ```text
/// ┌─────────────────────────────────────────┐
/// │  Tab bar (3 rows)                       │
/// ├─────────────────────────────────────────┤
/// │  Register table (fills remaining space) │
/// ├─────────────────────────────────────────┤
/// │  Status / filter bar (3 rows)           │
/// └─────────────────────────────────────────┘
/// ```
pub fn render(frame: &mut Frame, state: &BrowserState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // tab bar
            Constraint::Min(1),    // register table
            Constraint::Length(3), // status / filter bar
        ])
        .split(frame.area());

    render_tabs(frame, state, chunks[0]);
    render_table(frame, state, chunks[1]);
    render_status_bar(frame, state, chunks[2]);

    if state.show_help {
        render_help_overlay(frame);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tab bar
// ──────────────────────────────────────────────────────────────────────────────

fn render_tabs(frame: &mut Frame, state: &BrowserState, area: Rect) {
    let titles: Vec<Line<'_>> = RegisterTab::all()
        .iter()
        .map(|t| Line::from(t.title()))
        .collect();

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" OxiRS Modbus Browser "),
        )
        .select(state.active_tab.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        );

    frame.render_widget(tabs, area);
}

// ──────────────────────────────────────────────────────────────────────────────
// Register table
// ──────────────────────────────────────────────────────────────────────────────

fn render_table(frame: &mut Frame, state: &BrowserState, area: Rect) {
    let filtered = state.filtered_registers();

    let header_style = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let selected_style = Style::default()
        .bg(Color::DarkGray)
        .add_modifier(Modifier::BOLD);
    let error_style = Style::default().fg(Color::Red);
    let normal_style = Style::default().fg(Color::White);

    let header = Row::new(vec![
        Cell::from("Address").style(header_style),
        Cell::from("Value").style(header_style),
        Cell::from("Hex").style(header_style),
        Cell::from("Label").style(header_style),
        Cell::from("Status").style(header_style),
    ])
    .height(1);

    let rows: Vec<Row<'_>> = filtered
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let addr_str = format!("{:05}", entry.address);
            let value_str = entry.value.display();
            let hex_str = match &entry.value {
                RegisterValue::U16(v) => format!("0x{:04X}", v),
                RegisterValue::Bool(b) => if *b { "0x0001" } else { "0x0000" }.to_string(),
                RegisterValue::F32(_) => "IEEE754".to_string(),
                RegisterValue::Unknown => "—".to_string(),
            };
            let label_str = entry.label.as_deref().unwrap_or("—").to_string();
            let status_str = if let Some(err) = &entry.error {
                format!("ERR: {err}")
            } else {
                "OK".to_string()
            };

            let row_style = if entry.error.is_some() {
                error_style
            } else if i == state.selected_row {
                selected_style
            } else {
                normal_style
            };

            Row::new(vec![
                Cell::from(addr_str),
                Cell::from(value_str),
                Cell::from(hex_str),
                Cell::from(label_str),
                Cell::from(status_str),
            ])
            .style(row_style)
            .height(1)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),  // address
            Constraint::Length(12), // value
            Constraint::Length(10), // hex
            Constraint::Min(20),    // label
            Constraint::Length(20), // status
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(format!(
        " {} ({} entries) ",
        state.active_tab.title(),
        filtered.len()
    )))
    .row_highlight_style(selected_style);

    frame.render_widget(table, area);
}

// ──────────────────────────────────────────────────────────────────────────────
// Status / filter bar
// ──────────────────────────────────────────────────────────────────────────────

fn render_status_bar(frame: &mut Frame, state: &BrowserState, area: Rect) {
    let content = if state.edit_mode {
        let entry_addr = state
            .selected_entry()
            .map(|e| format!("0x{:04X}", e.address))
            .unwrap_or_else(|| "—".to_string());
        format!(" EDIT [{}]  Value: {}█", entry_addr, state.edit_buffer)
    } else if !state.filter.is_empty() {
        format!(" FILTER: {}█  (Enter=confirm  Esc=clear)", state.filter)
    } else if let Some(msg) = &state.status_message {
        format!(" {msg}")
    } else {
        " q=Quit  Tab=Next tab  j/k=Scroll  /=Filter  e=Edit  r=Refresh  ?=Help".to_string()
    };

    let style = if state.edit_mode {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else if !state.filter.is_empty() {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let bar = Paragraph::new(Line::from(Span::styled(content, style)))
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Left);

    frame.render_widget(bar, area);
}

// ──────────────────────────────────────────────────────────────────────────────
// Help overlay
// ──────────────────────────────────────────────────────────────────────────────

fn render_help_overlay(frame: &mut Frame) {
    let area = frame.area();
    // Centre a box that is 60% wide and 80% tall.
    let popup = centered_rect(60, 80, area);
    frame.render_widget(Clear, popup);
    frame.render_widget(help_overlay_widget(), popup);
}

/// Compute a centred rectangle of `percent_x` × `percent_y` of `area`.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}
