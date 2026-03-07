//! Kinematic equations for linear and rotational motion.
//!
//! Implements constant-acceleration kinematics for both linear
//! (translational) and rotational motion, plus projectile motion helpers.

/// Kinematic state for linear motion under constant acceleration.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearState {
    /// Position in meters.
    pub position: f64,
    /// Velocity in m/s.
    pub velocity: f64,
    /// Constant acceleration in m/s².
    pub acceleration: f64,
}

/// Kinematic state for rotational motion under constant angular acceleration.
#[derive(Debug, Clone, PartialEq)]
pub struct RotationalState {
    /// Angular position in radians.
    pub angle: f64,
    /// Angular velocity in rad/s.
    pub omega: f64,
    /// Constant angular acceleration in rad/s².
    pub alpha: f64,
}

/// Solver for classical kinematic equations (constant acceleration).
pub struct Kinematics;

impl Default for Kinematics {
    fn default() -> Self {
        Self::new()
    }
}

impl Kinematics {
    /// Create a new `Kinematics` solver.
    pub fn new() -> Self {
        Kinematics
    }

    // -----------------------------------------------------------------------
    // Linear kinematics
    // -----------------------------------------------------------------------

    /// Velocity at time `t`: v = v₀ + a·t
    pub fn velocity_at(&self, state: &LinearState, t: f64) -> f64 {
        state.velocity + state.acceleration * t
    }

    /// Position at time `t`: x = x₀ + v₀·t + ½·a·t²
    pub fn position_at(&self, state: &LinearState, t: f64) -> f64 {
        state.position + state.velocity * t + 0.5 * state.acceleration * t * t
    }

    /// Velocity after displacement `dx` using v² = v₀² + 2·a·Δx.
    ///
    /// Returns `Some(v)` with the non-negative root when the discriminant is
    /// non-negative (taking the sign that matches the direction of motion when
    /// possible), or `None` when the discriminant is negative.
    pub fn velocity_from_displacement(&self, state: &LinearState, dx: f64) -> Option<f64> {
        let discriminant = state.velocity * state.velocity + 2.0 * state.acceleration * dx;
        if discriminant < 0.0 {
            return None;
        }
        Some(discriminant.sqrt())
    }

    /// Time to reach velocity `v` from `state.velocity` under constant `a`.
    ///
    /// Returns `None` when `a = 0` and `v ≠ state.velocity` (unreachable).
    /// Returns `Some(0.0)` when `v == state.velocity`.
    pub fn time_to_velocity(&self, state: &LinearState, v: f64) -> Option<f64> {
        let delta_v = v - state.velocity;
        if state.acceleration.abs() < f64::EPSILON {
            if delta_v.abs() < f64::EPSILON {
                return Some(0.0);
            }
            return None;
        }
        Some(delta_v / state.acceleration)
    }

    /// Propagate the linear state forward by time `t`.
    pub fn propagate_linear(&self, state: &LinearState, t: f64) -> LinearState {
        LinearState {
            position: self.position_at(state, t),
            velocity: self.velocity_at(state, t),
            acceleration: state.acceleration,
        }
    }

    // -----------------------------------------------------------------------
    // Projectile motion
    // -----------------------------------------------------------------------

    /// Horizontal range of a projectile: R = v₀² · sin(2θ) / g
    pub fn projectile_range(&self, v0: f64, angle_rad: f64, g: f64) -> f64 {
        v0 * v0 * (2.0 * angle_rad).sin() / g
    }

    /// Maximum height of a projectile: H = (v₀ · sin θ)² / (2g)
    pub fn projectile_max_height(&self, v0: f64, angle_rad: f64, g: f64) -> f64 {
        let vy = v0 * angle_rad.sin();
        vy * vy / (2.0 * g)
    }

    /// Total time of flight: T = 2 · v₀ · sin θ / g
    pub fn projectile_time_of_flight(&self, v0: f64, angle_rad: f64, g: f64) -> f64 {
        2.0 * v0 * angle_rad.sin() / g
    }

    // -----------------------------------------------------------------------
    // Rotational kinematics
    // -----------------------------------------------------------------------

    /// Angular velocity at time `t`: ω = ω₀ + α·t
    pub fn angular_velocity_at(&self, state: &RotationalState, t: f64) -> f64 {
        state.omega + state.alpha * t
    }

    /// Angle at time `t`: θ = θ₀ + ω₀·t + ½·α·t²
    pub fn angle_at(&self, state: &RotationalState, t: f64) -> f64 {
        state.angle + state.omega * t + 0.5 * state.alpha * t * t
    }

    /// Propagate the rotational state forward by time `t`.
    pub fn propagate_rotational(&self, state: &RotationalState, t: f64) -> RotationalState {
        RotationalState {
            angle: self.angle_at(state, t),
            omega: self.angular_velocity_at(state, t),
            alpha: state.alpha,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_4, FRAC_PI_6, PI};

    const EPS: f64 = 1e-10;
    const G: f64 = 9.81;

    fn kin() -> Kinematics {
        Kinematics::new()
    }

    fn linear(pos: f64, vel: f64, acc: f64) -> LinearState {
        LinearState {
            position: pos,
            velocity: vel,
            acceleration: acc,
        }
    }

    fn rotational(angle: f64, omega: f64, alpha: f64) -> RotationalState {
        RotationalState {
            angle,
            omega,
            alpha,
        }
    }

    // -----------------------------------------------------------------------
    // velocity_at
    // -----------------------------------------------------------------------

    #[test]
    fn test_velocity_at_zero_time() {
        let k = kin();
        let s = linear(0.0, 5.0, 2.0);
        assert!((k.velocity_at(&s, 0.0) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_velocity_at_positive_time() {
        let k = kin();
        let s = linear(0.0, 0.0, 10.0);
        // v = 0 + 10 * 3 = 30
        assert!((k.velocity_at(&s, 3.0) - 30.0).abs() < EPS);
    }

    #[test]
    fn test_velocity_at_deceleration() {
        let k = kin();
        let s = linear(0.0, 20.0, -5.0);
        // v = 20 - 5*2 = 10
        assert!((k.velocity_at(&s, 2.0) - 10.0).abs() < EPS);
    }

    #[test]
    fn test_velocity_at_zero_acceleration() {
        let k = kin();
        let s = linear(100.0, 15.0, 0.0);
        assert!((k.velocity_at(&s, 7.0) - 15.0).abs() < EPS);
    }

    // -----------------------------------------------------------------------
    // position_at
    // -----------------------------------------------------------------------

    #[test]
    fn test_position_at_zero_time() {
        let k = kin();
        let s = linear(50.0, 3.0, 2.0);
        assert!((k.position_at(&s, 0.0) - 50.0).abs() < EPS);
    }

    #[test]
    fn test_position_at_constant_velocity() {
        let k = kin();
        let s = linear(0.0, 10.0, 0.0);
        // x = 10 * 5 = 50
        assert!((k.position_at(&s, 5.0) - 50.0).abs() < EPS);
    }

    #[test]
    fn test_position_at_with_acceleration() {
        let k = kin();
        let s = linear(0.0, 0.0, 2.0);
        // x = 0.5 * 2 * 4² = 16
        assert!((k.position_at(&s, 4.0) - 16.0).abs() < EPS);
    }

    #[test]
    fn test_position_at_full_equation() {
        let k = kin();
        let s = linear(10.0, 5.0, 2.0);
        // x = 10 + 5*3 + 0.5*2*9 = 10 + 15 + 9 = 34
        assert!((k.position_at(&s, 3.0) - 34.0).abs() < EPS);
    }

    // -----------------------------------------------------------------------
    // propagate_linear
    // -----------------------------------------------------------------------

    #[test]
    fn test_propagate_linear_preserves_acceleration() {
        let k = kin();
        let s = linear(0.0, 5.0, 3.0);
        let next = k.propagate_linear(&s, 2.0);
        assert!((next.acceleration - 3.0).abs() < EPS);
    }

    #[test]
    fn test_propagate_linear_position() {
        let k = kin();
        let s = linear(0.0, 0.0, 2.0);
        let next = k.propagate_linear(&s, 3.0);
        // x = 0.5 * 2 * 9 = 9
        assert!((next.position - 9.0).abs() < EPS);
    }

    #[test]
    fn test_propagate_linear_velocity() {
        let k = kin();
        let s = linear(0.0, 5.0, 4.0);
        let next = k.propagate_linear(&s, 2.0);
        assert!((next.velocity - 13.0).abs() < EPS);
    }

    #[test]
    fn test_propagate_linear_zero_time_identity() {
        let k = kin();
        let s = linear(7.0, 3.0, -1.5);
        let next = k.propagate_linear(&s, 0.0);
        assert!((next.position - s.position).abs() < EPS);
        assert!((next.velocity - s.velocity).abs() < EPS);
    }

    // -----------------------------------------------------------------------
    // velocity_from_displacement
    // -----------------------------------------------------------------------

    #[test]
    fn test_velocity_from_displacement_basic() {
        let k = kin();
        let s = linear(0.0, 0.0, 10.0);
        // v² = 0 + 2*10*5 = 100  → v = 10
        let v = k.velocity_from_displacement(&s, 5.0).expect("ok");
        assert!((v - 10.0).abs() < EPS);
    }

    #[test]
    fn test_velocity_from_displacement_negative_discriminant() {
        let k = kin();
        // v₀ = 1, a = -10, dx = 100 → v² = 1 - 2000 < 0
        let s = linear(0.0, 1.0, -10.0);
        assert!(k.velocity_from_displacement(&s, 100.0).is_none());
    }

    #[test]
    fn test_velocity_from_displacement_zero() {
        let k = kin();
        let s = linear(0.0, 3.0, 0.0);
        // v² = 9 → v = 3
        let v = k.velocity_from_displacement(&s, 0.0).expect("ok");
        assert!((v - 3.0).abs() < EPS);
    }

    #[test]
    fn test_velocity_from_displacement_returns_positive() {
        let k = kin();
        let s = linear(0.0, 4.0, 0.0);
        let v = k.velocity_from_displacement(&s, 0.0).expect("ok");
        assert!(v >= 0.0);
    }

    // -----------------------------------------------------------------------
    // time_to_velocity
    // -----------------------------------------------------------------------

    #[test]
    fn test_time_to_velocity_basic() {
        let k = kin();
        let s = linear(0.0, 0.0, 5.0);
        // t = (25 - 0) / 5 = 5
        let t = k.time_to_velocity(&s, 25.0).expect("ok");
        assert!((t - 5.0).abs() < EPS);
    }

    #[test]
    fn test_time_to_velocity_already_there() {
        let k = kin();
        let s = linear(0.0, 10.0, 3.0);
        let t = k.time_to_velocity(&s, 10.0).expect("ok");
        assert!((t).abs() < EPS);
    }

    #[test]
    fn test_time_to_velocity_zero_accel_same_vel() {
        let k = kin();
        let s = linear(0.0, 5.0, 0.0);
        let t = k.time_to_velocity(&s, 5.0);
        assert_eq!(t, Some(0.0));
    }

    #[test]
    fn test_time_to_velocity_zero_accel_different_vel() {
        let k = kin();
        let s = linear(0.0, 5.0, 0.0);
        // Cannot reach v=10 with zero acceleration
        assert!(k.time_to_velocity(&s, 10.0).is_none());
    }

    #[test]
    fn test_time_to_velocity_deceleration() {
        let k = kin();
        let s = linear(0.0, 20.0, -4.0);
        // t = (0 - 20) / (-4) = 5
        let t = k.time_to_velocity(&s, 0.0).expect("ok");
        assert!((t - 5.0).abs() < EPS);
    }

    // -----------------------------------------------------------------------
    // Projectile motion – 45° launch angle
    // -----------------------------------------------------------------------

    #[test]
    fn test_projectile_range_45_degrees() {
        let k = kin();
        let v0 = 20.0;
        let range = k.projectile_range(v0, FRAC_PI_4, G);
        // R = v₀² / g  at 45°
        let expected = v0 * v0 / G;
        assert!((range - expected).abs() < 1e-9);
    }

    #[test]
    fn test_projectile_max_height_45_degrees() {
        let k = kin();
        let v0 = 10.0;
        let h = k.projectile_max_height(v0, FRAC_PI_4, G);
        // H = (v₀/√2)² / (2g) = v₀²/(4g)
        let expected = v0 * v0 / (4.0 * G);
        assert!((h - expected).abs() < 1e-9);
    }

    #[test]
    fn test_projectile_time_of_flight_45_degrees() {
        let k = kin();
        let v0 = 10.0;
        let tof = k.projectile_time_of_flight(v0, FRAC_PI_4, G);
        let expected = 2.0 * v0 * FRAC_PI_4.sin() / G;
        assert!((tof - expected).abs() < 1e-9);
    }

    #[test]
    fn test_projectile_range_90_degrees_zero() {
        // At 90° sin(2*90°) = sin(180°) = 0 → range = 0
        let k = kin();
        let range = k.projectile_range(10.0, PI / 2.0, G);
        assert!(range.abs() < 1e-9);
    }

    #[test]
    fn test_projectile_range_30_degrees() {
        let k = kin();
        let v0 = 30.0;
        let range = k.projectile_range(v0, FRAC_PI_6, G);
        // R = 900 * sin(60°) / 9.81
        let expected = 900.0 * (PI / 3.0).sin() / G;
        assert!((range - expected).abs() < 1e-6);
    }

    #[test]
    fn test_projectile_max_height_vertical() {
        let k = kin();
        let v0 = 10.0;
        // At 90°: H = v₀²/(2g)
        let h = k.projectile_max_height(v0, PI / 2.0, G);
        let expected = v0 * v0 / (2.0 * G);
        assert!((h - expected).abs() < 1e-9);
    }

    #[test]
    fn test_projectile_time_of_flight_positive() {
        let k = kin();
        let tof = k.projectile_time_of_flight(20.0, FRAC_PI_4, G);
        assert!(tof > 0.0);
    }

    // -----------------------------------------------------------------------
    // Rotational kinematics
    // -----------------------------------------------------------------------

    #[test]
    fn test_angular_velocity_at_zero_time() {
        let k = kin();
        let s = rotational(0.0, 3.0, 1.0);
        assert!((k.angular_velocity_at(&s, 0.0) - 3.0).abs() < EPS);
    }

    #[test]
    fn test_angular_velocity_at_positive_time() {
        let k = kin();
        let s = rotational(0.0, 2.0, 4.0);
        // ω = 2 + 4*3 = 14
        assert!((k.angular_velocity_at(&s, 3.0) - 14.0).abs() < EPS);
    }

    #[test]
    fn test_angle_at_zero_time() {
        let k = kin();
        let s = rotational(1.5, 2.0, 0.5);
        assert!((k.angle_at(&s, 0.0) - 1.5).abs() < EPS);
    }

    #[test]
    fn test_angle_at_positive_time() {
        let k = kin();
        let s = rotational(0.0, 0.0, 2.0);
        // θ = 0.5 * 2 * 4² = 0.5 * 2 * 16 = 16
        assert!((k.angle_at(&s, 4.0) - 16.0).abs() < EPS);
    }

    #[test]
    fn test_angle_at_full_equation() {
        let k = kin();
        let s = rotational(1.0, 3.0, 2.0);
        // θ = 1 + 3*2 + 0.5*2*4 = 1 + 6 + 4 = 11
        assert!((k.angle_at(&s, 2.0) - 11.0).abs() < EPS);
    }

    #[test]
    fn test_propagate_rotational_preserves_alpha() {
        let k = kin();
        let s = rotational(0.0, 1.0, 2.0);
        let next = k.propagate_rotational(&s, 3.0);
        assert!((next.alpha - 2.0).abs() < EPS);
    }

    #[test]
    fn test_propagate_rotational_angle() {
        let k = kin();
        let s = rotational(0.0, 0.0, 2.0);
        let next = k.propagate_rotational(&s, 3.0);
        // θ = 0.5 * 2 * 9 = 9
        assert!((next.angle - 9.0).abs() < EPS);
    }

    #[test]
    fn test_propagate_rotational_omega() {
        let k = kin();
        let s = rotational(0.0, 1.0, 3.0);
        let next = k.propagate_rotational(&s, 4.0);
        // ω = 1 + 3*4 = 13
        assert!((next.omega - 13.0).abs() < EPS);
    }

    #[test]
    fn test_propagate_rotational_zero_time_identity() {
        let k = kin();
        let s = rotational(2.5, 1.0, -0.5);
        let next = k.propagate_rotational(&s, 0.0);
        assert!((next.angle - s.angle).abs() < EPS);
        assert!((next.omega - s.omega).abs() < EPS);
    }

    // -----------------------------------------------------------------------
    // Consistency between linear and rotational (mirror)
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_rotational_analogy() {
        let k = kin();
        // Linear with same parameters should mirror rotational
        let lin = linear(1.0, 3.0, 0.5);
        let rot = rotational(1.0, 3.0, 0.5);
        assert!((k.position_at(&lin, 2.0) - k.angle_at(&rot, 2.0)).abs() < EPS);
        assert!((k.velocity_at(&lin, 2.0) - k.angular_velocity_at(&rot, 2.0)).abs() < EPS);
    }

    // -----------------------------------------------------------------------
    // Default
    // -----------------------------------------------------------------------

    #[test]
    fn test_default() {
        let k = Kinematics;
        let s = linear(0.0, 5.0, 0.0);
        assert!((k.velocity_at(&s, 1.0) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_linear_state_clone() {
        let s = linear(1.0, 2.0, 3.0);
        let c = s.clone();
        assert_eq!(s, c);
    }

    #[test]
    fn test_rotational_state_clone() {
        let s = rotational(0.1, 0.2, 0.3);
        let c = s.clone();
        assert_eq!(s, c);
    }
}
