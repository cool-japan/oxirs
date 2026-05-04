//! Demo of the OxiRS Modbus TUI register browser.
//!
//! Loads a synthetic snapshot of all four register spaces — no real Modbus
//! device required — and launches the interactive browser.
//!
//! # Run
//!
//! ```bash
//! cargo run --example tui_browser_demo --features tui
//! ```
//!
//! # Key bindings
//!
//! | Key | Action |
//! |-----|--------|
//! | `Tab` / `Shift+Tab` | Switch tabs |
//! | `j` / `k` | Scroll down / up |
//! | `/` | Filter registers |
//! | `e` | Edit a holding register value |
//! | `r` | Refresh timestamps |
//! | `?` | Toggle help |
//! | `q` | Quit |

#[cfg(not(feature = "tui"))]
fn main() {
    eprintln!("This example requires the `tui` feature. Run with --features tui");
}

#[cfg(feature = "tui")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oxirs_modbus::browser::{
        state::{RegisterTab, RegisterValue},
        BrowserConfig, RegisterBrowser,
    };

    let config = BrowserConfig {
        poll_interval_ms: 200,
        max_address: 50,
        device_id: 1,
    };

    let mut browser = RegisterBrowser::new(config);

    // ── Coils (FC01) ─────────────────────────────────────────────────────────
    browser.load_snapshot(
        RegisterTab::Coils,
        vec![
            (0, RegisterValue::Bool(true), Some("Pump 1 Run".to_string())),
            (
                1,
                RegisterValue::Bool(false),
                Some("Pump 2 Run".to_string()),
            ),
            (
                2,
                RegisterValue::Bool(true),
                Some("Valve A Open".to_string()),
            ),
            (
                3,
                RegisterValue::Bool(false),
                Some("Valve B Open".to_string()),
            ),
            (
                4,
                RegisterValue::Bool(true),
                Some("Heater Enable".to_string()),
            ),
            (
                5,
                RegisterValue::Bool(false),
                Some("Alarm Reset".to_string()),
            ),
            (
                6,
                RegisterValue::Bool(true),
                Some("System Ready".to_string()),
            ),
            (
                7,
                RegisterValue::Bool(false),
                Some("Emergency Stop".to_string()),
            ),
        ],
    );

    // ── Discrete Inputs (FC02) ────────────────────────────────────────────────
    browser.load_snapshot(
        RegisterTab::DiscreteInputs,
        vec![
            (
                0,
                RegisterValue::Bool(true),
                Some("Door Sensor".to_string()),
            ),
            (
                1,
                RegisterValue::Bool(false),
                Some("Level High".to_string()),
            ),
            (2, RegisterValue::Bool(true), Some("Level Low".to_string())),
            (3, RegisterValue::Bool(false), Some("Flow OK".to_string())),
            (
                4,
                RegisterValue::Bool(true),
                Some("Pressure High".to_string()),
            ),
        ],
    );

    // ── Holding Registers (FC03) ──────────────────────────────────────────────
    browser.load_snapshot(
        RegisterTab::HoldingRegisters,
        vec![
            (
                0,
                RegisterValue::U16(1500),
                Some("Motor Speed RPM".to_string()),
            ),
            (
                1,
                RegisterValue::F32(23.75),
                Some("Temperature Setpoint C".to_string()),
            ),
            (
                2,
                RegisterValue::U16(4095),
                Some("DAC Output (12-bit)".to_string()),
            ),
            (
                3,
                RegisterValue::U16(0),
                Some("Control Mode (0=Manual)".to_string()),
            ),
            (
                4,
                RegisterValue::U16(100),
                Some("PID P-Gain x100".to_string()),
            ),
            (
                5,
                RegisterValue::U16(10),
                Some("PID I-Gain x100".to_string()),
            ),
            (
                6,
                RegisterValue::U16(5),
                Some("PID D-Gain x100".to_string()),
            ),
            (
                7,
                RegisterValue::U16(3600),
                Some("Setpoint Timeout s".to_string()),
            ),
            (8, RegisterValue::F32(0.5), Some("Deadband C".to_string())),
            (
                9,
                RegisterValue::U16(65535),
                Some("Alarm Mask Bits".to_string()),
            ),
            (
                10,
                RegisterValue::U16(240),
                Some("Supply Voltage V x10".to_string()),
            ),
            (
                11,
                RegisterValue::U16(150),
                Some("Current mA x10".to_string()),
            ),
        ],
    );

    // ── Input Registers (FC04) ────────────────────────────────────────────────
    browser.load_snapshot(
        RegisterTab::InputRegisters,
        vec![
            (
                0,
                RegisterValue::F32(22.3),
                Some("Temperature C".to_string()),
            ),
            (
                2,
                RegisterValue::F32(1013.25),
                Some("Pressure hPa".to_string()),
            ),
            (
                4,
                RegisterValue::U16(512),
                Some("ADC Channel 0 (12-bit)".to_string()),
            ),
            (
                5,
                RegisterValue::U16(1024),
                Some("ADC Channel 1 (12-bit)".to_string()),
            ),
            (
                6,
                RegisterValue::U16(2048),
                Some("ADC Channel 2 (12-bit)".to_string()),
            ),
            (
                7,
                RegisterValue::U16(0),
                Some("ADC Channel 3 (12-bit)".to_string()),
            ),
            (8, RegisterValue::U16(50), Some("Fan Speed %".to_string())),
            (9, RegisterValue::U16(0), Some("Error Code".to_string())),
        ],
    );

    // Launch the TUI.
    browser.run()?;

    Ok(())
}
