//! CAN bus bit timing calculations.
//!
//! Computes baud rate prescalers, time segment values (TSEG1, TSEG2, SJW),
//! sample point optimisation, oscillator tolerance, bit timing register
//! configuration for SJA1000 and MCP2515 controllers, CAN FD arbitration
//! and data phase timing, and valid configuration search.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// CAN protocol variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanVariant {
    /// Classical CAN 2.0 (up to 1 Mbit/s).
    Can20,
    /// CAN FD arbitration phase.
    CanFdArbitration,
    /// CAN FD data phase (up to 8 Mbit/s).
    CanFdData,
}

/// CAN controller type for register encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControllerType {
    /// Philips SJA1000.
    Sja1000,
    /// Microchip MCP2515.
    Mcp2515,
    /// Generic (BTR0/BTR1 encoding not applicable).
    Generic,
}

/// A complete bit timing configuration.
#[derive(Debug, Clone)]
pub struct BitTimingConfig {
    /// Clock frequency of the CAN controller (Hz).
    pub clock_hz: u64,
    /// Target baud rate (bit/s).
    pub baud_rate: u64,
    /// Baud rate prescaler (BRP). The time quantum (TQ) is `1 / (clock_hz / brp)`.
    pub brp: u32,
    /// Propagation segment (in TQ). Part of TSEG1 in most controllers.
    pub prop_seg: u8,
    /// Phase segment 1 (in TQ).
    pub phase_seg1: u8,
    /// Phase segment 2 (in TQ).
    pub phase_seg2: u8,
    /// Synchronisation Jump Width (in TQ).
    pub sjw: u8,
    /// Total number of time quanta per bit.
    pub total_tq: u8,
    /// Actual sample point percentage (0.0 - 100.0).
    pub sample_point: f64,
    /// Actual achieved baud rate (bit/s).
    pub actual_baud_rate: f64,
    /// Baud rate error percentage.
    pub baud_error_pct: f64,
}

/// CAN FD timing configuration (both phases).
#[derive(Debug, Clone)]
pub struct CanFdTimingConfig {
    /// Arbitration phase timing.
    pub arbitration: BitTimingConfig,
    /// Data phase timing.
    pub data: BitTimingConfig,
    /// Transceiver delay compensation value (TQ), if applicable.
    pub tdc_offset: Option<u8>,
}

/// SJA1000 BTR register pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sja1000Registers {
    /// Bus Timing Register 0: SJW\[7:6\], BRP\[5:0\].
    pub btr0: u8,
    /// Bus Timing Register 1: SAM\[7\], TSEG2\[6:4\], TSEG1\[3:0\].
    pub btr1: u8,
}

/// MCP2515 register set (CNF1, CNF2, CNF3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mcp2515Registers {
    /// CNF1: SJW\[7:6\], BRP\[5:0\].
    pub cnf1: u8,
    /// CNF2: BTLMODE\[7\], SAM\[6\], PHSEG1\[5:3\], PRSEG\[2:0\].
    pub cnf2: u8,
    /// CNF3: SOF\[7\], WAKFIL\[6\], (unused\[5:3\]), PHSEG2\[2:0\].
    pub cnf3: u8,
}

/// Constraints for configuration search.
#[derive(Debug, Clone)]
pub struct TimingConstraints {
    /// Target sample point percentage (typical: 75.0 - 87.5 for CAN 2.0).
    pub target_sample_point: f64,
    /// Acceptable sample point tolerance (percentage points).
    pub sample_point_tolerance: f64,
    /// Maximum BRP value.
    pub max_brp: u32,
    /// Maximum total TQ per bit.
    pub max_tq: u8,
    /// Minimum total TQ per bit.
    pub min_tq: u8,
    /// Maximum acceptable baud rate error (percentage).
    pub max_baud_error_pct: f64,
}

impl Default for TimingConstraints {
    fn default() -> Self {
        Self {
            target_sample_point: 87.5,
            sample_point_tolerance: 5.0,
            max_brp: 64,
            max_tq: 25,
            min_tq: 8,
            max_baud_error_pct: 0.5,
        }
    }
}

/// Oscillator tolerance analysis result.
#[derive(Debug, Clone)]
pub struct OscillatorTolerance {
    /// Maximum tolerable clock deviation (ppm).
    pub max_deviation_ppm: f64,
    /// Whether the configuration meets the CAN specification requirement.
    pub meets_spec: bool,
    /// Minimum oscillator accuracy required.
    pub required_accuracy_pct: f64,
}

/// Errors from bit timing calculations.
#[derive(Debug)]
pub enum TimingError {
    /// No valid configuration found.
    NoValidConfig(String),
    /// Baud rate is not achievable with the given clock.
    InvalidBaudRate(u64),
    /// Clock frequency is too low.
    ClockTooLow(u64),
    /// BRP value out of range.
    BrpOutOfRange(u32),
    /// Invalid parameter.
    InvalidParameter(String),
}

impl std::fmt::Display for TimingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimingError::NoValidConfig(msg) => write!(f, "no valid config: {msg}"),
            TimingError::InvalidBaudRate(br) => write!(f, "invalid baud rate: {br}"),
            TimingError::ClockTooLow(clk) => write!(f, "clock too low: {clk} Hz"),
            TimingError::BrpOutOfRange(brp) => write!(f, "BRP out of range: {brp}"),
            TimingError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for TimingError {}

// ─────────────────────────────────────────────────────────────────────────────
// BitTimingCalculator
// ─────────────────────────────────────────────────────────────────────────────

/// CAN bus bit timing calculator.
pub struct BitTimingCalculator {
    /// Clock frequency in Hz.
    clock_hz: u64,
    /// Search constraints.
    constraints: TimingConstraints,
    /// Cache of previously computed configs: (baud_rate) -> config.
    cache: HashMap<u64, BitTimingConfig>,
}

impl BitTimingCalculator {
    /// Create a new calculator for the given clock frequency.
    pub fn new(clock_hz: u64, constraints: TimingConstraints) -> Self {
        Self {
            clock_hz,
            constraints,
            cache: HashMap::new(),
        }
    }

    /// Create with default constraints.
    pub fn with_defaults(clock_hz: u64) -> Self {
        Self::new(clock_hz, TimingConstraints::default())
    }

    /// Get the clock frequency.
    pub fn clock_hz(&self) -> u64 {
        self.clock_hz
    }

    /// Get the constraints.
    pub fn constraints(&self) -> &TimingConstraints {
        &self.constraints
    }

    /// Set new constraints.
    pub fn set_constraints(&mut self, constraints: TimingConstraints) {
        self.constraints = constraints;
        self.cache.clear();
    }

    // ─── Core Calculations ───────────────────────────────────────────────

    /// Compute the time quantum (TQ) duration in nanoseconds for a given BRP.
    pub fn tq_ns(&self, brp: u32) -> f64 {
        if self.clock_hz == 0 {
            return 0.0;
        }
        (brp as f64 / self.clock_hz as f64) * 1_000_000_000.0
    }

    /// Compute the baud rate from BRP and total TQ per bit.
    pub fn compute_baud_rate(&self, brp: u32, total_tq: u8) -> f64 {
        if brp == 0 || total_tq == 0 {
            return 0.0;
        }
        self.clock_hz as f64 / (brp as f64 * total_tq as f64)
    }

    /// Compute the sample point percentage.
    ///
    /// Sample point = (1 + prop_seg + phase_seg1) / total_tq * 100
    pub fn compute_sample_point(prop_seg: u8, phase_seg1: u8, total_tq: u8) -> f64 {
        if total_tq == 0 {
            return 0.0;
        }
        (1.0 + prop_seg as f64 + phase_seg1 as f64) / total_tq as f64 * 100.0
    }

    /// Compute BRP from clock and target baud rate for a given total TQ.
    pub fn compute_brp(&self, baud_rate: u64, total_tq: u8) -> Option<u32> {
        if baud_rate == 0 || total_tq == 0 {
            return None;
        }
        let brp = self.clock_hz as f64 / (baud_rate as f64 * total_tq as f64);
        let brp_rounded = brp.round() as u32;
        if brp_rounded == 0 || brp_rounded > self.constraints.max_brp {
            return None;
        }
        Some(brp_rounded)
    }

    /// Compute the baud rate error percentage.
    pub fn baud_error(&self, target: u64, brp: u32, total_tq: u8) -> f64 {
        let actual = self.compute_baud_rate(brp, total_tq);
        if target == 0 {
            return 0.0;
        }
        ((actual - target as f64) / target as f64 * 100.0).abs()
    }

    // ─── Configuration Search ────────────────────────────────────────────

    /// Find the best bit timing configuration for a target baud rate.
    ///
    /// Searches all valid (BRP, TQ) combinations and selects the one with
    /// the closest sample point to the target.
    pub fn calculate(&mut self, baud_rate: u64) -> Result<BitTimingConfig, TimingError> {
        if baud_rate == 0 {
            return Err(TimingError::InvalidBaudRate(0));
        }
        if self.clock_hz < baud_rate {
            return Err(TimingError::ClockTooLow(self.clock_hz));
        }

        // Check cache
        if let Some(cached) = self.cache.get(&baud_rate) {
            return Ok(cached.clone());
        }

        let mut best: Option<BitTimingConfig> = None;
        let mut best_score = f64::MAX;

        for total_tq in self.constraints.min_tq..=self.constraints.max_tq {
            let brp = match self.compute_brp(baud_rate, total_tq) {
                Some(b) => b,
                None => continue,
            };

            let baud_err = self.baud_error(baud_rate, brp, total_tq);
            if baud_err > self.constraints.max_baud_error_pct {
                continue;
            }

            // Try different TSEG1/TSEG2 splits
            // total_tq = 1 (sync) + prop_seg + phase_seg1 + phase_seg2
            let available = total_tq.saturating_sub(1); // subtract sync segment
            if available < 3 {
                continue;
            }

            for phase_seg2 in 1..=(available / 2).min(8) {
                let tseg1_total = available - phase_seg2;
                if tseg1_total < 2 {
                    continue;
                }

                // Split TSEG1 into prop_seg and phase_seg1.
                // Typical: prop_seg >= 1, phase_seg1 >= 1.
                let prop_seg = (tseg1_total / 2).max(1);
                let phase_seg1 = tseg1_total - prop_seg;
                if phase_seg1 == 0 {
                    continue;
                }

                let sample_pt = Self::compute_sample_point(prop_seg, phase_seg1, total_tq);
                let sp_diff = (sample_pt - self.constraints.target_sample_point).abs();

                if sp_diff > self.constraints.sample_point_tolerance {
                    continue;
                }

                // SJW: min(4, phase_seg2)
                let sjw = phase_seg2.clamp(1, 4);

                let score = sp_diff + baud_err * 10.0;

                if score < best_score {
                    best_score = score;
                    let actual_baud = self.compute_baud_rate(brp, total_tq);
                    best = Some(BitTimingConfig {
                        clock_hz: self.clock_hz,
                        baud_rate,
                        brp,
                        prop_seg,
                        phase_seg1,
                        phase_seg2,
                        sjw,
                        total_tq,
                        sample_point: sample_pt,
                        actual_baud_rate: actual_baud,
                        baud_error_pct: baud_err,
                    });
                }
            }
        }

        match best {
            Some(config) => {
                self.cache.insert(baud_rate, config.clone());
                Ok(config)
            }
            None => Err(TimingError::NoValidConfig(format!(
                "no config for {baud_rate} bps with clock {0} Hz",
                self.clock_hz
            ))),
        }
    }

    /// Find all valid configurations for a baud rate.
    pub fn find_all_configs(&self, baud_rate: u64) -> Vec<BitTimingConfig> {
        if baud_rate == 0 || self.clock_hz < baud_rate {
            return Vec::new();
        }

        let mut configs = Vec::new();

        for total_tq in self.constraints.min_tq..=self.constraints.max_tq {
            let brp = match self.compute_brp(baud_rate, total_tq) {
                Some(b) => b,
                None => continue,
            };

            let baud_err = self.baud_error(baud_rate, brp, total_tq);
            if baud_err > self.constraints.max_baud_error_pct {
                continue;
            }

            let available = total_tq.saturating_sub(1);
            if available < 3 {
                continue;
            }

            for phase_seg2 in 1..=(available / 2).min(8) {
                let tseg1_total = available - phase_seg2;
                if tseg1_total < 2 {
                    continue;
                }

                let prop_seg = (tseg1_total / 2).max(1);
                let phase_seg1 = tseg1_total - prop_seg;
                if phase_seg1 == 0 {
                    continue;
                }

                let sample_pt = Self::compute_sample_point(prop_seg, phase_seg1, total_tq);
                let sp_diff = (sample_pt - self.constraints.target_sample_point).abs();
                if sp_diff > self.constraints.sample_point_tolerance {
                    continue;
                }

                let sjw = phase_seg2.clamp(1, 4);
                let actual_baud = self.compute_baud_rate(brp, total_tq);

                configs.push(BitTimingConfig {
                    clock_hz: self.clock_hz,
                    baud_rate,
                    brp,
                    prop_seg,
                    phase_seg1,
                    phase_seg2,
                    sjw,
                    total_tq,
                    sample_point: sample_pt,
                    actual_baud_rate: actual_baud,
                    baud_error_pct: baud_err,
                });
            }
        }

        configs
    }

    // ─── CAN FD Timing ──────────────────────────────────────────────────

    /// Calculate CAN FD timing for both arbitration and data phases.
    pub fn calculate_canfd(
        &mut self,
        arb_baud: u64,
        data_baud: u64,
    ) -> Result<CanFdTimingConfig, TimingError> {
        // Arbitration phase: standard CAN 2.0 constraints (87.5% sample point).
        let arb = self.calculate(arb_baud)?;

        // Data phase: typically lower sample point (~75%) and fewer TQ.
        let saved = self.constraints.clone();
        self.constraints.target_sample_point = 75.0;
        self.constraints.sample_point_tolerance = 12.5;
        self.constraints.min_tq = 5;
        self.constraints.max_tq = 15;
        let data = self.calculate(data_baud);
        self.constraints = saved;

        let data = data?;

        // TDC offset: typically phase_seg1 + prop_seg of the data phase.
        let tdc = data.prop_seg + data.phase_seg1;

        Ok(CanFdTimingConfig {
            arbitration: arb,
            data,
            tdc_offset: Some(tdc),
        })
    }

    // ─── Register Encoding ──────────────────────────────────────────────

    /// Encode a configuration as SJA1000 BTR0/BTR1 registers.
    pub fn encode_sja1000(config: &BitTimingConfig) -> Result<Sja1000Registers, TimingError> {
        let brp = config.brp.saturating_sub(1);
        if brp > 63 {
            return Err(TimingError::BrpOutOfRange(config.brp));
        }
        let sjw = config.sjw.saturating_sub(1).min(3);

        let btr0 = (sjw << 6) | (brp as u8 & 0x3F);

        let tseg1 = (config.prop_seg + config.phase_seg1).saturating_sub(1);
        let tseg2 = config.phase_seg2.saturating_sub(1);
        let btr1 = ((tseg2 & 0x07) << 4) | (tseg1 & 0x0F);

        Ok(Sja1000Registers { btr0, btr1 })
    }

    /// Encode a configuration as MCP2515 CNF1/CNF2/CNF3 registers.
    pub fn encode_mcp2515(config: &BitTimingConfig) -> Result<Mcp2515Registers, TimingError> {
        let brp = config.brp.saturating_sub(1);
        if brp > 63 {
            return Err(TimingError::BrpOutOfRange(config.brp));
        }
        let sjw = config.sjw.saturating_sub(1).min(3);

        let cnf1 = (sjw << 6) | (brp as u8 & 0x3F);

        let prseg = config.prop_seg.saturating_sub(1).min(7);
        let phseg1 = config.phase_seg1.saturating_sub(1).min(7);
        // BTLMODE=1 (bit 7), SAM=0 (bit 6)
        let cnf2 = 0x80 | ((phseg1 & 0x07) << 3) | (prseg & 0x07);

        let phseg2 = config.phase_seg2.saturating_sub(1).min(7);
        let cnf3 = phseg2 & 0x07;

        Ok(Mcp2515Registers { cnf1, cnf2, cnf3 })
    }

    // ─── Oscillator Tolerance ────────────────────────────────────────────

    /// Compute the maximum tolerable oscillator deviation.
    ///
    /// Based on CAN specification: the oscillator tolerance is bounded by
    /// `min(phase_seg1, phase_seg2) / (2 * (13 * total_tq - phase_seg2))`.
    pub fn oscillator_tolerance(config: &BitTimingConfig) -> OscillatorTolerance {
        let ps1 = config.phase_seg1 as f64;
        let ps2 = config.phase_seg2 as f64;
        let tq = config.total_tq as f64;
        let sjw = config.sjw as f64;

        // CAN spec formula (simplified):
        // df_max = min(phase_seg1, phase_seg2) / (2 * (13 * total_tq - phase_seg2))
        let numerator = ps1.min(ps2);
        let denominator = 2.0 * (13.0 * tq - ps2);

        let tolerance = if denominator.abs() > f64::EPSILON {
            numerator / denominator
        } else {
            0.0
        };

        // Also bounded by SJW:
        // df_max <= SJW / (20 * total_tq)
        let sjw_bound = if tq > 0.0 { sjw / (20.0 * tq) } else { 0.0 };

        let max_tolerance = tolerance.min(sjw_bound);
        let max_ppm = max_tolerance * 1_000_000.0;

        // CAN spec requires ≤ 1.58% for CAN 2.0 (most common is 0.5%)
        let meets_spec = max_tolerance >= 0.005; // 0.5%

        OscillatorTolerance {
            max_deviation_ppm: max_ppm,
            meets_spec,
            required_accuracy_pct: (1.0 - max_tolerance) * 100.0,
        }
    }

    // ─── Bit Time Verification ───────────────────────────────────────────

    /// Verify a configuration against CAN specification requirements.
    pub fn verify(config: &BitTimingConfig) -> Vec<String> {
        let mut issues = Vec::new();

        // Total TQ must be at least 8
        if config.total_tq < 8 {
            issues.push(format!("total_tq {} < 8 minimum", config.total_tq));
        }

        // SJW <= min(4, phase_seg2)
        if config.sjw > config.phase_seg2 {
            issues.push(format!(
                "SJW {} > phase_seg2 {}",
                config.sjw, config.phase_seg2
            ));
        }
        if config.sjw > 4 {
            issues.push(format!("SJW {} > 4 maximum", config.sjw));
        }

        // Sample point: 75-87.5% for CAN 2.0 (informational)
        if config.sample_point < 50.0 || config.sample_point > 95.0 {
            issues.push(format!(
                "sample point {:.1}% outside 50-95% range",
                config.sample_point
            ));
        }

        // Baud rate error
        if config.baud_error_pct > 1.0 {
            issues.push(format!(
                "baud rate error {:.2}% > 1.0%",
                config.baud_error_pct
            ));
        }

        // phase_seg1 >= 1, phase_seg2 >= 1
        if config.phase_seg1 == 0 {
            issues.push("phase_seg1 is 0".to_string());
        }
        if config.phase_seg2 == 0 {
            issues.push("phase_seg2 is 0".to_string());
        }

        issues
    }

    /// Clear the calculation cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Number of cached configurations.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Standard baud rates
// ─────────────────────────────────────────────────────────────────────────────

/// Standard CAN 2.0 baud rates.
pub const CAN_BAUD_10K: u64 = 10_000;
pub const CAN_BAUD_20K: u64 = 20_000;
pub const CAN_BAUD_50K: u64 = 50_000;
pub const CAN_BAUD_100K: u64 = 100_000;
pub const CAN_BAUD_125K: u64 = 125_000;
pub const CAN_BAUD_250K: u64 = 250_000;
pub const CAN_BAUD_500K: u64 = 500_000;
pub const CAN_BAUD_800K: u64 = 800_000;
pub const CAN_BAUD_1M: u64 = 1_000_000;

/// Standard CAN FD data phase baud rates.
pub const CANFD_BAUD_2M: u64 = 2_000_000;
pub const CANFD_BAUD_4M: u64 = 4_000_000;
pub const CANFD_BAUD_5M: u64 = 5_000_000;
pub const CANFD_BAUD_8M: u64 = 8_000_000;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const CLK_8MHZ: u64 = 8_000_000;
    const CLK_16MHZ: u64 = 16_000_000;
    const CLK_80MHZ: u64 = 80_000_000;

    fn make_calc(clock: u64) -> BitTimingCalculator {
        BitTimingCalculator::with_defaults(clock)
    }

    // ── Basic Computation Tests ──────────────────────────────────────────

    #[test]
    fn test_tq_ns() {
        let calc = make_calc(CLK_8MHZ);
        let tq = calc.tq_ns(1);
        assert!((tq - 125.0).abs() < 0.1); // 1/8MHz = 125ns
    }

    #[test]
    fn test_tq_ns_zero_clock() {
        let calc = BitTimingCalculator::with_defaults(0);
        assert!((calc.tq_ns(1) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_baud_rate() {
        let calc = make_calc(CLK_16MHZ);
        let baud = calc.compute_baud_rate(2, 8);
        // 16MHz / (2 * 8) = 1MHz
        assert!((baud - 1_000_000.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_baud_rate_zero() {
        let calc = make_calc(CLK_16MHZ);
        assert!((calc.compute_baud_rate(0, 8) - 0.0).abs() < f64::EPSILON);
        assert!((calc.compute_baud_rate(2, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_sample_point() {
        // total_tq=16, prop=4, phase1=4 => (1+4+4)/16*100 = 56.25%
        let sp = BitTimingCalculator::compute_sample_point(4, 4, 16);
        assert!((sp - 56.25).abs() < 0.01);
    }

    #[test]
    fn test_compute_sample_point_zero_tq() {
        assert!((BitTimingCalculator::compute_sample_point(4, 4, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_brp() {
        let calc = make_calc(CLK_16MHZ);
        // 16MHz / (500k * 16) = 2
        let brp = calc.compute_brp(500_000, 16);
        assert_eq!(brp, Some(2));
    }

    #[test]
    fn test_compute_brp_zero_baud() {
        let calc = make_calc(CLK_16MHZ);
        assert_eq!(calc.compute_brp(0, 16), None);
    }

    #[test]
    fn test_baud_error() {
        let calc = make_calc(CLK_16MHZ);
        let err = calc.baud_error(500_000, 2, 16);
        assert!(err < 0.01);
    }

    // ── Configuration Search Tests ───────────────────────────────────────

    #[test]
    fn test_calculate_500k() {
        let mut calc = make_calc(CLK_16MHZ);
        let config = calc.calculate(CAN_BAUD_500K);
        assert!(config.is_ok());
        let c = config.expect("config");
        assert!(c.baud_error_pct < 1.0);
        assert!(c.sample_point > 75.0);
    }

    #[test]
    fn test_calculate_250k() {
        let mut calc = make_calc(CLK_16MHZ);
        let config = calc.calculate(CAN_BAUD_250K);
        assert!(config.is_ok());
    }

    #[test]
    fn test_calculate_125k() {
        let mut calc = make_calc(CLK_8MHZ);
        let config = calc.calculate(CAN_BAUD_125K);
        assert!(config.is_ok());
    }

    #[test]
    fn test_calculate_1m() {
        let mut calc = make_calc(CLK_16MHZ);
        let config = calc.calculate(CAN_BAUD_1M);
        assert!(config.is_ok());
    }

    #[test]
    fn test_calculate_zero_baud_error() {
        let mut calc = make_calc(CLK_16MHZ);
        assert!(calc.calculate(0).is_err());
    }

    #[test]
    fn test_calculate_clock_too_low() {
        let mut calc = make_calc(100);
        assert!(calc.calculate(CAN_BAUD_1M).is_err());
    }

    #[test]
    fn test_calculate_caching() {
        let mut calc = make_calc(CLK_16MHZ);
        calc.calculate(CAN_BAUD_500K).ok();
        assert_eq!(calc.cache_size(), 1);
        calc.calculate(CAN_BAUD_500K).ok(); // should use cache
        assert_eq!(calc.cache_size(), 1);
    }

    #[test]
    fn test_clear_cache() {
        let mut calc = make_calc(CLK_16MHZ);
        calc.calculate(CAN_BAUD_500K).ok();
        calc.clear_cache();
        assert_eq!(calc.cache_size(), 0);
    }

    #[test]
    fn test_find_all_configs() {
        let calc = make_calc(CLK_16MHZ);
        let configs = calc.find_all_configs(CAN_BAUD_500K);
        assert!(!configs.is_empty());
    }

    #[test]
    fn test_find_all_configs_zero_baud() {
        let calc = make_calc(CLK_16MHZ);
        let configs = calc.find_all_configs(0);
        assert!(configs.is_empty());
    }

    // ── CAN FD Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_canfd_timing() {
        let mut calc = make_calc(CLK_80MHZ);
        let config = calc.calculate_canfd(CAN_BAUD_500K, CANFD_BAUD_2M);
        assert!(config.is_ok());
        let c = config.expect("canfd config");
        assert!(c.tdc_offset.is_some());
    }

    #[test]
    fn test_canfd_arb_baud() {
        let mut calc = make_calc(CLK_80MHZ);
        let config = calc
            .calculate_canfd(CAN_BAUD_500K, CANFD_BAUD_2M)
            .expect("config");
        assert!(config.arbitration.baud_error_pct < 1.0);
    }

    // ── Register Encoding Tests ──────────────────────────────────────────

    #[test]
    fn test_encode_sja1000() {
        let config = BitTimingConfig {
            clock_hz: CLK_16MHZ,
            baud_rate: CAN_BAUD_500K,
            brp: 2,
            prop_seg: 3,
            phase_seg1: 4,
            phase_seg2: 3,
            sjw: 1,
            total_tq: 11,
            sample_point: 72.7,
            actual_baud_rate: 500_000.0,
            baud_error_pct: 0.0,
        };
        let regs = BitTimingCalculator::encode_sja1000(&config);
        assert!(regs.is_ok());
        let r = regs.expect("sja1000 registers");
        // BRP-1 = 1, SJW-1 = 0 => BTR0 = 0x01
        assert_eq!(r.btr0 & 0x3F, 1);
    }

    #[test]
    fn test_encode_sja1000_brp_out_of_range() {
        let config = BitTimingConfig {
            clock_hz: CLK_16MHZ,
            baud_rate: 10,
            brp: 100,
            prop_seg: 3,
            phase_seg1: 4,
            phase_seg2: 3,
            sjw: 1,
            total_tq: 11,
            sample_point: 72.7,
            actual_baud_rate: 10.0,
            baud_error_pct: 0.0,
        };
        assert!(BitTimingCalculator::encode_sja1000(&config).is_err());
    }

    #[test]
    fn test_encode_mcp2515() {
        let config = BitTimingConfig {
            clock_hz: CLK_16MHZ,
            baud_rate: CAN_BAUD_500K,
            brp: 2,
            prop_seg: 3,
            phase_seg1: 4,
            phase_seg2: 3,
            sjw: 1,
            total_tq: 11,
            sample_point: 72.7,
            actual_baud_rate: 500_000.0,
            baud_error_pct: 0.0,
        };
        let regs = BitTimingCalculator::encode_mcp2515(&config);
        assert!(regs.is_ok());
        let r = regs.expect("mcp2515 registers");
        // BTLMODE bit should be set
        assert_ne!(r.cnf2 & 0x80, 0);
    }

    #[test]
    fn test_encode_mcp2515_brp_out_of_range() {
        let config = BitTimingConfig {
            clock_hz: CLK_16MHZ,
            baud_rate: 10,
            brp: 100,
            prop_seg: 3,
            phase_seg1: 4,
            phase_seg2: 3,
            sjw: 1,
            total_tq: 11,
            sample_point: 72.7,
            actual_baud_rate: 10.0,
            baud_error_pct: 0.0,
        };
        assert!(BitTimingCalculator::encode_mcp2515(&config).is_err());
    }

    // ── Oscillator Tolerance Tests ───────────────────────────────────────

    #[test]
    fn test_oscillator_tolerance() {
        let config = BitTimingConfig {
            clock_hz: CLK_16MHZ,
            baud_rate: CAN_BAUD_500K,
            brp: 2,
            prop_seg: 3,
            phase_seg1: 4,
            phase_seg2: 3,
            sjw: 1,
            total_tq: 11,
            sample_point: 72.7,
            actual_baud_rate: 500_000.0,
            baud_error_pct: 0.0,
        };
        let tol = BitTimingCalculator::oscillator_tolerance(&config);
        assert!(tol.max_deviation_ppm > 0.0);
    }

    #[test]
    fn test_oscillator_tolerance_accuracy() {
        let config = BitTimingConfig {
            clock_hz: CLK_16MHZ,
            baud_rate: CAN_BAUD_500K,
            brp: 2,
            prop_seg: 3,
            phase_seg1: 4,
            phase_seg2: 3,
            sjw: 1,
            total_tq: 11,
            sample_point: 72.7,
            actual_baud_rate: 500_000.0,
            baud_error_pct: 0.0,
        };
        let tol = BitTimingCalculator::oscillator_tolerance(&config);
        assert!(tol.required_accuracy_pct > 0.0);
        assert!(tol.required_accuracy_pct < 100.0);
    }

    // ── Verification Tests ───────────────────────────────────────────────

    #[test]
    fn test_verify_valid_config() {
        let mut calc = make_calc(CLK_16MHZ);
        let config = calc.calculate(CAN_BAUD_500K).expect("config");
        let issues = BitTimingCalculator::verify(&config);
        assert!(issues.is_empty(), "issues: {:?}", issues);
    }

    #[test]
    fn test_verify_low_total_tq() {
        let config = BitTimingConfig {
            clock_hz: CLK_16MHZ,
            baud_rate: CAN_BAUD_500K,
            brp: 2,
            prop_seg: 1,
            phase_seg1: 1,
            phase_seg2: 1,
            sjw: 1,
            total_tq: 4,
            sample_point: 75.0,
            actual_baud_rate: 500_000.0,
            baud_error_pct: 0.0,
        };
        let issues = BitTimingCalculator::verify(&config);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_verify_sjw_exceeds_phase_seg2() {
        let config = BitTimingConfig {
            clock_hz: CLK_16MHZ,
            baud_rate: CAN_BAUD_500K,
            brp: 2,
            prop_seg: 3,
            phase_seg1: 4,
            phase_seg2: 1,
            sjw: 3,
            total_tq: 12,
            sample_point: 75.0,
            actual_baud_rate: 500_000.0,
            baud_error_pct: 0.0,
        };
        let issues = BitTimingCalculator::verify(&config);
        assert!(issues.iter().any(|i| i.contains("SJW")));
    }

    // ── Constraints Tests ────────────────────────────────────────────────

    #[test]
    fn test_default_constraints() {
        let c = TimingConstraints::default();
        assert!((c.target_sample_point - 87.5).abs() < f64::EPSILON);
        assert_eq!(c.max_brp, 64);
    }

    #[test]
    fn test_set_constraints() {
        let mut calc = make_calc(CLK_16MHZ);
        calc.calculate(CAN_BAUD_500K).ok();
        assert_eq!(calc.cache_size(), 1);
        let c = TimingConstraints {
            target_sample_point: 75.0,
            ..TimingConstraints::default()
        };
        calc.set_constraints(c);
        assert_eq!(calc.cache_size(), 0); // cache cleared
    }

    // ── Accessors Tests ──────────────────────────────────────────────────

    #[test]
    fn test_clock_accessor() {
        let calc = make_calc(CLK_16MHZ);
        assert_eq!(calc.clock_hz(), CLK_16MHZ);
    }

    #[test]
    fn test_constraints_accessor() {
        let calc = make_calc(CLK_16MHZ);
        assert!((calc.constraints().target_sample_point - 87.5).abs() < f64::EPSILON);
    }

    // ── Error Display Tests ──────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = TimingError::NoValidConfig("test".to_string());
        assert!(format!("{e}").contains("test"));
        let e = TimingError::InvalidBaudRate(999);
        assert!(format!("{e}").contains("999"));
        let e = TimingError::ClockTooLow(100);
        assert!(format!("{e}").contains("100"));
        let e = TimingError::BrpOutOfRange(99);
        assert!(format!("{e}").contains("99"));
        let e = TimingError::InvalidParameter("bad".to_string());
        assert!(format!("{e}").contains("bad"));
    }

    // ── Standard Baud Rate Constants Tests ───────────────────────────────

    #[test]
    fn test_standard_baud_rates() {
        assert_eq!(CAN_BAUD_125K, 125_000);
        assert_eq!(CAN_BAUD_250K, 250_000);
        assert_eq!(CAN_BAUD_500K, 500_000);
        assert_eq!(CAN_BAUD_1M, 1_000_000);
        assert_eq!(CANFD_BAUD_2M, 2_000_000);
        assert_eq!(CANFD_BAUD_5M, 5_000_000);
    }

    // ── 80MHz Clock Tests ────────────────────────────────────────────────

    #[test]
    fn test_calculate_500k_80mhz() {
        let mut calc = make_calc(CLK_80MHZ);
        let config = calc.calculate(CAN_BAUD_500K);
        assert!(config.is_ok());
        let c = config.expect("config");
        assert!(c.baud_error_pct < 0.5);
    }

    #[test]
    fn test_calculate_1m_80mhz() {
        let mut calc = make_calc(CLK_80MHZ);
        let config = calc.calculate(CAN_BAUD_1M);
        assert!(config.is_ok());
    }

    // ── Register Equality Tests ──────────────────────────────────────────

    #[test]
    fn test_sja1000_register_equality() {
        let r1 = Sja1000Registers {
            btr0: 0x01,
            btr1: 0x1C,
        };
        let r2 = Sja1000Registers {
            btr0: 0x01,
            btr1: 0x1C,
        };
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_mcp2515_register_equality() {
        let r1 = Mcp2515Registers {
            cnf1: 0x01,
            cnf2: 0x90,
            cnf3: 0x02,
        };
        let r2 = Mcp2515Registers {
            cnf1: 0x01,
            cnf2: 0x90,
            cnf3: 0x02,
        };
        assert_eq!(r1, r2);
    }
}
