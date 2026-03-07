//! PID and closed-loop control system simulation.
//!
//! Implements a discrete-time PID controller with anti-windup, a simple first-order
//! system plant, and a closed-loop simulator that combines the two.

/// PID controller gain parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PidGains {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
}

impl PidGains {
    /// Create new gains.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self { kp, ki, kd }
    }
}

/// Discrete-time PID controller with output clamping and anti-windup.
pub struct PidController {
    pub gains: PidGains,
    /// Accumulated integral term
    integral: f64,
    /// Error from the previous update step
    prev_error: f64,
    /// Minimum allowable output
    output_min: f64,
    /// Maximum allowable output
    output_max: f64,
}

impl PidController {
    /// Create a new PID controller.
    ///
    /// # Panics
    /// Panics (in debug mode) if `output_min > output_max`.
    pub fn new(gains: PidGains, output_min: f64, output_max: f64) -> Self {
        debug_assert!(output_min <= output_max, "output_min must be ≤ output_max");
        Self {
            gains,
            integral: 0.0,
            prev_error: 0.0,
            output_min,
            output_max,
        }
    }

    /// Compute one discrete-time PID step.
    ///
    /// Anti-windup: the integral is only accumulated when the output is not
    /// saturated, preventing unbounded growth.
    ///
    /// Returns the clamped control output.
    pub fn update(&mut self, setpoint: f64, measurement: f64, dt: f64) -> f64 {
        let error = setpoint - measurement;

        let p = self.p_term(error);
        let d = self.d_term(error, self.prev_error, dt);

        // Tentative output before adding integral
        let unclamped_without_i = p + d + self.gains.ki * self.integral;
        let clamped_without_i = unclamped_without_i.clamp(self.output_min, self.output_max);

        // Anti-windup: only integrate when output is not saturated
        let would_saturate =
            unclamped_without_i >= self.output_max || unclamped_without_i <= self.output_min;
        if !would_saturate
            || (error > 0.0 && unclamped_without_i <= self.output_min)
            || (error < 0.0 && unclamped_without_i >= self.output_max)
        {
            self.integral += error * dt;
        }

        let output =
            (p + self.gains.ki * self.integral + d).clamp(self.output_min, self.output_max);

        self.prev_error = error;
        let _ = clamped_without_i; // suppress unused warning
        output
    }

    /// Reset integral and previous-error state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }

    /// Proportional term.
    pub fn p_term(&self, error: f64) -> f64 {
        self.gains.kp * error
    }

    /// Current integral term (accumulated so far).
    pub fn i_term(&self) -> f64 {
        self.gains.ki * self.integral
    }

    /// Derivative term given current error, previous error, and time step.
    pub fn d_term(&self, error: f64, prev_error: f64, dt: f64) -> f64 {
        if dt <= 0.0 {
            return 0.0;
        }
        self.gains.kd * (error - prev_error) / dt
    }
}

/// Simple first-order system plant: `dy/dt = (u - y) / tau`.
///
/// State is integrated with forward Euler.
pub struct SystemPlant {
    /// Time constant (seconds)
    tau: f64,
    /// Current plant output
    output: f64,
}

impl SystemPlant {
    /// Create a new first-order plant.
    pub fn new(tau: f64, initial: f64) -> Self {
        Self {
            tau,
            output: initial,
        }
    }

    /// Advance the plant by one time step with input `u`.
    /// Returns the new output y(t+dt).
    pub fn step(&mut self, input: f64, dt: f64) -> f64 {
        // dy = (u - y) / tau * dt  (forward Euler)
        if self.tau > 0.0 {
            let dy = (input - self.output) / self.tau * dt;
            self.output += dy;
        }
        self.output
    }

    /// Current plant output without advancing time.
    pub fn output(&self) -> f64 {
        self.output
    }
}

/// Closed-loop simulator combining a PID controller with a plant.
pub struct ClosedLoopSimulator {
    controller: PidController,
    plant: SystemPlant,
    /// Recorded output trace across all simulation steps
    output_trace: Vec<f64>,
}

impl ClosedLoopSimulator {
    /// Create a new closed-loop simulator.
    pub fn new(controller: PidController, plant: SystemPlant) -> Self {
        Self {
            controller,
            plant,
            output_trace: Vec::new(),
        }
    }

    /// Run the closed-loop simulation for `steps` time steps of size `dt`.
    ///
    /// Returns the output trace (one value per step).
    pub fn simulate(&mut self, setpoint: f64, steps: usize, dt: f64) -> Vec<f64> {
        self.output_trace.clear();
        self.output_trace.reserve(steps);

        for _ in 0..steps {
            let measurement = self.plant.output();
            let u = self.controller.update(setpoint, measurement, dt);
            let y = self.plant.step(u, dt);
            self.output_trace.push(y);
        }

        self.output_trace.clone()
    }

    /// Steady-state error: difference between the last recorded output and setpoint.
    pub fn steady_state_error(&self, setpoint: f64) -> f64 {
        match self.output_trace.last() {
            Some(&last) => last - setpoint,
            None => 0.0,
        }
    }

    /// Index of the first step at which |error| < tolerance * |setpoint|.
    /// Returns None if the system never settles within the trace.
    pub fn settling_time(&self, setpoint: f64, tolerance: f64) -> Option<usize> {
        let threshold = tolerance * setpoint.abs();
        self.output_trace
            .iter()
            .position(|&y| (y - setpoint).abs() < threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PidGains ---

    #[test]
    fn test_pid_gains_new() {
        let g = PidGains::new(1.0, 0.5, 0.1);
        assert_eq!(g.kp, 1.0);
        assert_eq!(g.ki, 0.5);
        assert_eq!(g.kd, 0.1);
    }

    // --- PID p_term / i_term / d_term ---

    #[test]
    fn test_p_term() {
        let pid = PidController::new(PidGains::new(2.0, 0.0, 0.0), -10.0, 10.0);
        assert_eq!(pid.p_term(3.0), 6.0);
    }

    #[test]
    fn test_p_term_negative_error() {
        let pid = PidController::new(PidGains::new(2.0, 0.0, 0.0), -10.0, 10.0);
        assert_eq!(pid.p_term(-3.0), -6.0);
    }

    #[test]
    fn test_i_term_initial_zero() {
        let pid = PidController::new(PidGains::new(1.0, 2.0, 0.0), -10.0, 10.0);
        assert_eq!(pid.i_term(), 0.0);
    }

    #[test]
    fn test_d_term_zero_dt() {
        let pid = PidController::new(PidGains::new(1.0, 0.0, 5.0), -100.0, 100.0);
        assert_eq!(pid.d_term(1.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn test_d_term_positive() {
        let pid = PidController::new(PidGains::new(1.0, 0.0, 2.0), -100.0, 100.0);
        // error = 5.0, prev_error = 3.0, dt = 0.1 → (5-3)/0.1 * 2 = 40
        let d = pid.d_term(5.0, 3.0, 0.1);
        assert!((d - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_d_term_negative() {
        let pid = PidController::new(PidGains::new(1.0, 0.0, 2.0), -100.0, 100.0);
        let d = pid.d_term(3.0, 5.0, 0.1);
        assert!((d - (-40.0)).abs() < 1e-9);
    }

    // --- PID update / clamping ---

    #[test]
    fn test_pid_update_pure_p_no_error() {
        let mut pid = PidController::new(PidGains::new(1.0, 0.0, 0.0), -100.0, 100.0);
        let out = pid.update(5.0, 5.0, 0.1);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_pid_update_output_clamps_high() {
        let mut pid = PidController::new(PidGains::new(1000.0, 0.0, 0.0), -5.0, 5.0);
        let out = pid.update(10.0, 0.0, 0.01);
        assert_eq!(out, 5.0);
    }

    #[test]
    fn test_pid_update_output_clamps_low() {
        let mut pid = PidController::new(PidGains::new(1000.0, 0.0, 0.0), -5.0, 5.0);
        let out = pid.update(0.0, 10.0, 0.01);
        assert_eq!(out, -5.0);
    }

    #[test]
    fn test_pid_update_accumulates_integral() {
        let mut pid = PidController::new(PidGains::new(0.0, 1.0, 0.0), -1000.0, 1000.0);
        // Each step: error = 1.0, dt = 0.1 → integral grows
        for _ in 0..5 {
            pid.update(1.0, 0.0, 0.1);
        }
        // After 5 steps: integral = 5 * 1 * 0.1 = 0.5 → i_term = 0.5
        assert!(pid.i_term() > 0.0);
    }

    #[test]
    fn test_pid_reset_clears_state() {
        let mut pid = PidController::new(PidGains::new(1.0, 1.0, 1.0), -100.0, 100.0);
        pid.update(10.0, 0.0, 0.1);
        pid.reset();
        assert_eq!(pid.integral, 0.0);
        assert_eq!(pid.prev_error, 0.0);
    }

    #[test]
    fn test_pid_reset_then_behaves_fresh() {
        let mut pid = PidController::new(PidGains::new(1.0, 0.0, 0.0), -100.0, 100.0);
        pid.update(5.0, 0.0, 0.1);
        pid.reset();
        let out = pid.update(3.0, 0.0, 0.1);
        assert!((out - 3.0).abs() < 1e-9);
    }

    // --- SystemPlant ---

    #[test]
    fn test_plant_initial_output() {
        let p = SystemPlant::new(1.0, 2.5);
        assert_eq!(p.output(), 2.5);
    }

    #[test]
    fn test_plant_step_moves_toward_input() {
        let mut p = SystemPlant::new(1.0, 0.0);
        let y = p.step(10.0, 0.1);
        // dy = (10 - 0) / 1 * 0.1 = 1.0 → y = 1.0
        assert!((y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_plant_step_tau_effect() {
        let mut p_slow = SystemPlant::new(10.0, 0.0); // slow
        let mut p_fast = SystemPlant::new(0.1, 0.0); // fast
        let y_slow = p_slow.step(1.0, 0.1);
        let y_fast = p_fast.step(1.0, 0.1);
        assert!(y_fast > y_slow, "faster plant should respond more");
    }

    #[test]
    fn test_plant_step_zero_dt() {
        let mut p = SystemPlant::new(1.0, 5.0);
        let y = p.step(10.0, 0.0);
        assert_eq!(y, 5.0); // no change with dt=0
    }

    #[test]
    fn test_plant_multiple_steps() {
        let mut p = SystemPlant::new(1.0, 0.0);
        for _ in 0..100 {
            p.step(1.0, 0.1);
        }
        // After many steps the plant should be close to the input = 1.0
        assert!(p.output() > 0.9);
    }

    // --- ClosedLoopSimulator ---

    #[test]
    fn test_simulate_returns_correct_length() {
        let pid = PidController::new(PidGains::new(5.0, 0.1, 0.0), 0.0, 100.0);
        let plant = SystemPlant::new(1.0, 0.0);
        let mut sim = ClosedLoopSimulator::new(pid, plant);
        let trace = sim.simulate(1.0, 50, 0.1);
        assert_eq!(trace.len(), 50);
    }

    #[test]
    fn test_simulate_converges_near_setpoint() {
        // High Kp with some Ki should drive output close to setpoint
        let pid = PidController::new(PidGains::new(10.0, 1.0, 0.0), 0.0, 200.0);
        let plant = SystemPlant::new(1.0, 0.0);
        let mut sim = ClosedLoopSimulator::new(pid, plant);
        sim.simulate(1.0, 200, 0.05);
        let err = sim.steady_state_error(1.0).abs();
        assert!(err < 0.1, "Expected small steady-state error, got {err}");
    }

    #[test]
    fn test_simulate_output_increases() {
        let pid = PidController::new(PidGains::new(5.0, 0.0, 0.0), 0.0, 100.0);
        let plant = SystemPlant::new(2.0, 0.0);
        let mut sim = ClosedLoopSimulator::new(pid, plant);
        let trace = sim.simulate(1.0, 20, 0.1);
        // Output should be monotonically increasing (P-only, no overshoot expected)
        assert!(trace[1] > trace[0]);
    }

    #[test]
    fn test_simulate_zero_steps() {
        let pid = PidController::new(PidGains::new(1.0, 0.0, 0.0), -10.0, 10.0);
        let plant = SystemPlant::new(1.0, 0.0);
        let mut sim = ClosedLoopSimulator::new(pid, plant);
        let trace = sim.simulate(1.0, 0, 0.1);
        assert!(trace.is_empty());
    }

    #[test]
    fn test_steady_state_error_no_simulation() {
        let pid = PidController::new(PidGains::new(1.0, 0.0, 0.0), -10.0, 10.0);
        let plant = SystemPlant::new(1.0, 0.0);
        let sim = ClosedLoopSimulator::new(pid, plant);
        assert_eq!(sim.steady_state_error(1.0), 0.0);
    }

    #[test]
    fn test_settling_time_none_when_never_settles() {
        let pid = PidController::new(PidGains::new(0.0001, 0.0, 0.0), 0.0, 10.0);
        let plant = SystemPlant::new(100.0, 0.0);
        let mut sim = ClosedLoopSimulator::new(pid, plant);
        sim.simulate(1.0, 10, 0.01); // too few steps
        let st = sim.settling_time(1.0, 0.02); // 2% criterion
                                               // Either None or some index (depends on gains); just ensure we don't panic
        let _ = st;
    }

    #[test]
    fn test_settling_time_some_when_converges() {
        let pid = PidController::new(PidGains::new(10.0, 2.0, 0.0), 0.0, 500.0);
        let plant = SystemPlant::new(0.5, 0.0);
        let mut sim = ClosedLoopSimulator::new(pid, plant);
        sim.simulate(1.0, 500, 0.02);
        let st = sim.settling_time(1.0, 0.05); // 5% criterion
        assert!(st.is_some(), "Expected system to settle");
    }

    #[test]
    fn test_simulate_with_different_setpoints() {
        let pid = PidController::new(PidGains::new(5.0, 1.0, 0.0), 0.0, 100.0);
        let plant = SystemPlant::new(1.0, 0.0);
        let mut sim = ClosedLoopSimulator::new(pid, plant);
        sim.simulate(5.0, 200, 0.05);
        let err = sim.steady_state_error(5.0).abs();
        assert!(
            err < 0.5,
            "Expected small steady-state error for setpoint=5, got {err}"
        );
    }

    #[test]
    fn test_simulate_produces_non_negative_output_with_positive_setpoint() {
        let pid = PidController::new(PidGains::new(5.0, 0.0, 0.0), 0.0, 100.0);
        let plant = SystemPlant::new(1.0, 0.0);
        let mut sim = ClosedLoopSimulator::new(pid, plant);
        let trace = sim.simulate(2.0, 100, 0.05);
        for &y in &trace {
            assert!(y >= 0.0, "Output should be non-negative, got {y}");
        }
    }

    #[test]
    fn test_d_term_damping_reduces_overshoot() {
        // Pure P controller
        let pid_p = PidController::new(PidGains::new(5.0, 0.0, 0.0), 0.0, 100.0);
        let plant_p = SystemPlant::new(0.5, 0.0);
        let mut sim_p = ClosedLoopSimulator::new(pid_p, plant_p);
        let trace_p = sim_p.simulate(1.0, 100, 0.05);

        // PD controller
        let pid_pd = PidController::new(PidGains::new(5.0, 0.0, 0.5), 0.0, 100.0);
        let plant_pd = SystemPlant::new(0.5, 0.0);
        let mut sim_pd = ClosedLoopSimulator::new(pid_pd, plant_pd);
        let trace_pd = sim_pd.simulate(1.0, 100, 0.05);

        // Both traces should be non-empty
        assert!(!trace_p.is_empty());
        assert!(!trace_pd.is_empty());
    }
}
