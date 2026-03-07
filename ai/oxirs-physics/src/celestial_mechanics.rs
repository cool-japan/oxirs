//! # Celestial Mechanics
//!
//! Orbital mechanics and N-body gravitational simulation based on Newtonian gravity.
//!
//! ## Features
//!
//! - **N-body simulator**: Euler-integrated gravitational dynamics for arbitrary numbers of bodies.
//! - **Energy tracking**: kinetic, potential, and total energy calculations.
//! - **Orbital calculator**: analytical formulae for circular-orbit velocity, orbital period
//!   (Kepler's third law), escape velocity, and vis-viva equation.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_physics::celestial_mechanics::{Body, NBodySimulator, OrbitalCalculator};
//!
//! // Sun–Earth simplified two-body system
//! let mut sim = NBodySimulator::new();
//! sim.add_body(Body {
//!     name: "Sun".to_string(),
//!     mass: 1.989e30,
//!     position: [0.0; 3],
//!     velocity: [0.0; 3],
//! });
//! sim.add_body(Body {
//!     name: "Earth".to_string(),
//!     mass: 5.972e24,
//!     position: [1.496e11, 0.0, 0.0],
//!     velocity: [0.0, 29_780.0, 0.0],
//! });
//! sim.step_n(60.0, 100); // 100 steps of 60 seconds each
//!
//! let v_circ = OrbitalCalculator::circular_orbit_velocity(1.989e30, 1.496e11);
//! assert!((v_circ - 29_780.0).abs() < 500.0);
//! ```

/// Gravitational constant (m³ kg⁻¹ s⁻²).
pub const G: f64 = 6.674e-11;

// ─── Body ─────────────────────────────────────────────────────────────────────

/// A point mass participating in the N-body simulation.
#[derive(Debug, Clone)]
pub struct Body {
    /// Human-readable name (e.g. `"Earth"`).
    pub name: String,
    /// Mass in kilograms.
    pub mass: f64,
    /// Position vector in metres `[x, y, z]`.
    pub position: [f64; 3],
    /// Velocity vector in metres per second `[vx, vy, vz]`.
    pub velocity: [f64; 3],
}

impl Body {
    /// Compute the kinetic energy: ½ m v².
    pub fn kinetic_energy(&self) -> f64 {
        let v_sq: f64 = self.velocity.iter().map(|v| v * v).sum();
        0.5 * self.mass * v_sq
    }
}

// ─── OrbitalElements ─────────────────────────────────────────────────────────

/// Classical Keplerian orbital elements (two-body problem).
#[derive(Debug, Clone)]
pub struct OrbitalElements {
    /// Length of the semi-major axis in metres.
    pub semi_major_axis: f64,
    /// Orbital eccentricity (0 = circular).
    pub eccentricity: f64,
    /// Inclination of the orbital plane in degrees.
    pub inclination_deg: f64,
    /// Orbital period in seconds.
    pub period_s: f64,
}

// ─── NBodySimulator ───────────────────────────────────────────────────────────

/// Simple N-body gravitational simulator using first-order Euler integration.
///
/// Suitable for qualitative demonstrations; for production accuracy use a
/// higher-order integrator (e.g. Runge-Kutta 4 or Leapfrog).
pub struct NBodySimulator {
    bodies: Vec<Body>,
}

impl NBodySimulator {
    /// Create an empty simulator with no bodies.
    pub fn new() -> Self {
        Self { bodies: Vec::new() }
    }

    /// Add a body to the simulation.
    pub fn add_body(&mut self, body: Body) {
        self.bodies.push(body);
    }

    /// Number of bodies currently in the simulation.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Retrieve a body by name, or `None` if not found.
    pub fn body(&self, name: &str) -> Option<&Body> {
        self.bodies.iter().find(|b| b.name == name)
    }

    /// Mutable access to a body by name.
    pub fn body_mut(&mut self, name: &str) -> Option<&mut Body> {
        self.bodies.iter_mut().find(|b| b.name == name)
    }

    /// Advance the simulation by one time step of `dt_s` seconds.
    ///
    /// Algorithm (Euler):
    /// 1. Compute the net gravitational force on each body from all others.
    /// 2. Update velocities: `v += (F/m) * dt`.
    /// 3. Update positions: `p += v_new * dt`.
    pub fn step(&mut self, dt_s: f64) {
        let n = self.bodies.len();
        // Accumulate accelerations.
        let mut accelerations: Vec<[f64; 3]> = vec![[0.0; 3]; n];

        for (i, acc_i) in accelerations.iter_mut().enumerate() {
            for (j, body_j) in self.bodies.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dx = body_j.position[0] - self.bodies[i].position[0];
                let dy = body_j.position[1] - self.bodies[i].position[1];
                let dz = body_j.position[2] - self.bodies[i].position[2];
                let r_sq = dx * dx + dy * dy + dz * dz;

                if r_sq < 1.0 {
                    // Avoid singularity for coincident bodies.
                    continue;
                }

                let r = r_sq.sqrt();
                // a_i += G * m_j / r² * r̂
                let factor = G * body_j.mass / (r_sq * r);
                acc_i[0] += factor * dx;
                acc_i[1] += factor * dy;
                acc_i[2] += factor * dz;
            }
        }

        // Apply Euler integration.
        for (body, acc) in self.bodies.iter_mut().zip(accelerations.iter()) {
            for (vel_k, acc_k) in body.velocity.iter_mut().zip(acc.iter()) {
                *vel_k += acc_k * dt_s;
            }
            for (pos_k, vel_k) in body.position.iter_mut().zip(body.velocity.iter()) {
                *pos_k += vel_k * dt_s;
            }
        }
    }

    /// Advance the simulation by `n` steps of `dt_s` seconds each.
    pub fn step_n(&mut self, dt_s: f64, n: usize) {
        for _ in 0..n {
            self.step(dt_s);
        }
    }

    /// Total kinetic energy of the system (sum of ½ mᵢ vᵢ²).
    pub fn kinetic_energy(&self) -> f64 {
        self.bodies.iter().map(|b| b.kinetic_energy()).sum()
    }

    /// Total gravitational potential energy of the system.
    ///
    /// Computed as the sum over all unique pairs (i, j):
    /// U = −G mᵢ mⱼ / |rᵢ − rⱼ|
    pub fn potential_energy(&self) -> f64 {
        let n = self.bodies.len();
        let mut u = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[i].position[0] - self.bodies[j].position[0];
                let dy = self.bodies[i].position[1] - self.bodies[j].position[1];
                let dz = self.bodies[i].position[2] - self.bodies[j].position[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r > 0.0 {
                    u -= G * self.bodies[i].mass * self.bodies[j].mass / r;
                }
            }
        }
        u
    }

    /// Total mechanical energy: kinetic + potential.
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Read-only slice of all bodies.
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// Compute the centre of mass position of the system.
    pub fn centre_of_mass(&self) -> [f64; 3] {
        let total_mass: f64 = self.bodies.iter().map(|b| b.mass).sum();
        if total_mass == 0.0 {
            return [0.0; 3];
        }
        let mut com = [0.0f64; 3];
        for body in &self.bodies {
            for (com_k, pos_k) in com.iter_mut().zip(body.position.iter()) {
                *com_k += body.mass * pos_k;
            }
        }
        for com_k in &mut com {
            *com_k /= total_mass;
        }
        com
    }

    /// Compute the total linear momentum of the system [kg m/s].
    pub fn total_momentum(&self) -> [f64; 3] {
        let mut p = [0.0f64; 3];
        for body in &self.bodies {
            for (p_k, vel_k) in p.iter_mut().zip(body.velocity.iter()) {
                *p_k += body.mass * vel_k;
            }
        }
        p
    }
}

impl Default for NBodySimulator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── OrbitalCalculator ───────────────────────────────────────────────────────

/// Analytical orbital mechanics formulae (Newtonian gravity).
pub struct OrbitalCalculator;

impl OrbitalCalculator {
    /// Circular orbit speed at distance `radius_m` from a body of mass `central_mass`.
    ///
    /// v = √(G M / r)
    pub fn circular_orbit_velocity(central_mass: f64, radius_m: f64) -> f64 {
        (G * central_mass / radius_m).sqrt()
    }

    /// Orbital period for an elliptical orbit via Kepler's third law.
    ///
    /// T = 2π √(a³ / (G M))
    pub fn orbital_period(central_mass: f64, semi_major_axis_m: f64) -> f64 {
        let a3 = semi_major_axis_m.powi(3);
        2.0 * std::f64::consts::PI * (a3 / (G * central_mass)).sqrt()
    }

    /// Escape velocity from the surface of a body of mass `central_mass`
    /// at distance `radius_m`.
    ///
    /// v_esc = √(2 G M / r)
    pub fn escape_velocity(central_mass: f64, radius_m: f64) -> f64 {
        (2.0 * G * central_mass / radius_m).sqrt()
    }

    /// Speed at distance `r` in an elliptical orbit with semi-major axis `a`
    /// around a body of mass `central_mass` (vis-viva equation).
    ///
    /// v² = G M (2/r − 1/a)
    ///
    /// Returns `f64::NAN` if the result would be negative (unbound orbit with
    /// invalid parameters).
    pub fn vis_viva(central_mass: f64, r: f64, a: f64) -> f64 {
        let v_sq = G * central_mass * (2.0 / r - 1.0 / a);
        if v_sq >= 0.0 {
            v_sq.sqrt()
        } else {
            f64::NAN
        }
    }

    /// Specific orbital energy (energy per unit mass) for an elliptical orbit.
    ///
    /// ε = −G M / (2 a)
    pub fn specific_orbital_energy(central_mass: f64, semi_major_axis_m: f64) -> f64 {
        -G * central_mass / (2.0 * semi_major_axis_m)
    }

    /// Compute the apoapsis distance given the periapsis distance and eccentricity.
    ///
    /// r_apo = r_peri * (1 + e) / (1 - e)
    pub fn apoapsis(periapsis_m: f64, eccentricity: f64) -> Option<f64> {
        if eccentricity >= 1.0 {
            None // hyperbolic or parabolic — no apoapsis
        } else {
            Some(periapsis_m * (1.0 + eccentricity) / (1.0 - eccentricity))
        }
    }

    /// Compute the periapsis distance given the apoapsis distance and eccentricity.
    pub fn periapsis(apoapsis_m: f64, eccentricity: f64) -> f64 {
        apoapsis_m * (1.0 - eccentricity) / (1.0 + eccentricity)
    }

    /// Semi-major axis from apoapsis and periapsis: a = (r_apo + r_peri) / 2.
    pub fn semi_major_axis(apoapsis_m: f64, periapsis_m: f64) -> f64 {
        (apoapsis_m + periapsis_m) / 2.0
    }

    /// Hill sphere radius — the region within which a body dominates the
    /// gravitational attraction over a parent body.
    ///
    /// r_H ≈ a (m / (3 M))^(1/3)
    pub fn hill_sphere(semi_major_axis_m: f64, body_mass: f64, parent_mass: f64) -> f64 {
        semi_major_axis_m * (body_mass / (3.0 * parent_mass)).cbrt()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── constants ────────────────────────────────────────────────────────────

    const SOLAR_MASS: f64 = 1.989e30; // kg
    const EARTH_MASS: f64 = 5.972e24; // kg
    const AU: f64 = 1.496e11; // m (1 astronomical unit)
    const EARTH_ORBITAL_V: f64 = 29_780.0; // m/s (approx circular)
    const ONE_YEAR_S: f64 = 3.156e7; // s

    // ── OrbitalCalculator: circular_orbit_velocity ────────────────────────────

    #[test]
    fn test_circular_orbit_velocity_earth() {
        let v = OrbitalCalculator::circular_orbit_velocity(SOLAR_MASS, AU);
        // Known value ~29,780 m/s; allow 1% error
        assert!(
            (v - EARTH_ORBITAL_V).abs() / EARTH_ORBITAL_V < 0.01,
            "circular velocity {v} deviates too much from {EARTH_ORBITAL_V}"
        );
    }

    #[test]
    fn test_circular_orbit_velocity_increases_with_mass() {
        let v1 = OrbitalCalculator::circular_orbit_velocity(SOLAR_MASS, AU);
        let v2 = OrbitalCalculator::circular_orbit_velocity(SOLAR_MASS * 4.0, AU);
        // v ∝ √M so v2 should be 2× v1
        assert!((v2 - 2.0 * v1).abs() / v1 < 1e-9);
    }

    #[test]
    fn test_circular_orbit_velocity_decreases_with_radius() {
        let v1 = OrbitalCalculator::circular_orbit_velocity(SOLAR_MASS, AU);
        let v2 = OrbitalCalculator::circular_orbit_velocity(SOLAR_MASS, AU * 4.0);
        // v ∝ 1/√r so v2 = v1 / 2
        assert!((v2 - v1 / 2.0).abs() / v1 < 1e-9);
    }

    #[test]
    fn test_circular_orbit_velocity_positive() {
        assert!(OrbitalCalculator::circular_orbit_velocity(1e20, 1e9) > 0.0);
    }

    // ── OrbitalCalculator: orbital_period ────────────────────────────────────

    #[test]
    fn test_orbital_period_earth_approx_one_year() {
        let t = OrbitalCalculator::orbital_period(SOLAR_MASS, AU);
        // Allow 2% relative error
        assert!(
            (t - ONE_YEAR_S).abs() / ONE_YEAR_S < 0.02,
            "period {t} s deviates too much from {ONE_YEAR_S} s"
        );
    }

    #[test]
    fn test_orbital_period_kepler_third_law_ratio() {
        // T² ∝ a³  =>  (T2/T1)² == (a2/a1)³
        let t1 = OrbitalCalculator::orbital_period(SOLAR_MASS, AU);
        let t2 = OrbitalCalculator::orbital_period(SOLAR_MASS, AU * 4.0);
        let ratio_t = t2 / t1;
        // a2 = 4 a1  =>  ratio_t = 4^(3/2) = 8
        assert!((ratio_t - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_orbital_period_positive() {
        assert!(OrbitalCalculator::orbital_period(1e20, 1e9) > 0.0);
    }

    // ── OrbitalCalculator: escape_velocity ───────────────────────────────────

    #[test]
    fn test_escape_velocity_is_sqrt2_times_circular() {
        let v_c = OrbitalCalculator::circular_orbit_velocity(SOLAR_MASS, AU);
        let v_e = OrbitalCalculator::escape_velocity(SOLAR_MASS, AU);
        assert!((v_e - (2.0_f64).sqrt() * v_c).abs() / v_c < 1e-9);
    }

    #[test]
    fn test_escape_velocity_earth_surface() {
        // Earth radius 6.371e6 m, known escape velocity ~11,186 m/s
        let v_esc = OrbitalCalculator::escape_velocity(EARTH_MASS, 6.371e6);
        assert!(
            (v_esc - 11_186.0).abs() / 11_186.0 < 0.01,
            "escape velocity {v_esc} m/s deviates too much from 11,186 m/s"
        );
    }

    #[test]
    fn test_escape_velocity_greater_than_circular() {
        let v_c = OrbitalCalculator::circular_orbit_velocity(SOLAR_MASS, AU);
        let v_e = OrbitalCalculator::escape_velocity(SOLAR_MASS, AU);
        assert!(v_e > v_c);
    }

    // ── OrbitalCalculator: vis_viva ───────────────────────────────────────────

    #[test]
    fn test_vis_viva_circular_equals_circular_velocity() {
        // For a circular orbit r == a, so vis-viva should give the circular speed.
        let v_vv = OrbitalCalculator::vis_viva(SOLAR_MASS, AU, AU);
        let v_c = OrbitalCalculator::circular_orbit_velocity(SOLAR_MASS, AU);
        assert!((v_vv - v_c).abs() / v_c < 1e-9);
    }

    #[test]
    fn test_vis_viva_at_periapsis() {
        // Periapsis of an ellipse with a=1.5 AU, e=0.5  =>  r_peri = a(1-e) = 0.75 AU
        let a = 1.5 * AU;
        let e = 0.5;
        let r_peri = a * (1.0 - e);
        let v = OrbitalCalculator::vis_viva(SOLAR_MASS, r_peri, a);
        assert!(v > 0.0);
    }

    #[test]
    fn test_vis_viva_at_apoapsis_less_than_periapsis_speed() {
        let a = 1.5 * AU;
        let e = 0.5;
        let r_peri = a * (1.0 - e);
        let r_apo = a * (1.0 + e);
        let v_peri = OrbitalCalculator::vis_viva(SOLAR_MASS, r_peri, a);
        let v_apo = OrbitalCalculator::vis_viva(SOLAR_MASS, r_apo, a);
        assert!(
            v_peri > v_apo,
            "speed at periapsis should be greater than at apoapsis"
        );
    }

    #[test]
    fn test_vis_viva_parabolic_orbit_at_peri_equals_escape() {
        // Parabolic orbit: a → ∞  =>  vis-viva → escape velocity
        // For a very large a the formula approaches v_esc
        let large_a = 1e30;
        let v_vv = OrbitalCalculator::vis_viva(SOLAR_MASS, AU, large_a);
        let v_esc = OrbitalCalculator::escape_velocity(SOLAR_MASS, AU);
        assert!((v_vv - v_esc).abs() / v_esc < 1e-3);
    }

    // ── OrbitalCalculator: apoapsis / periapsis ──────────────────────────────

    #[test]
    fn test_apoapsis_circular_orbit() {
        // e = 0: apoapsis == periapsis
        let apo = OrbitalCalculator::apoapsis(AU, 0.0).expect("circular has apoapsis");
        assert!((apo - AU).abs() < 1.0);
    }

    #[test]
    fn test_apoapsis_elliptical() {
        // e = 0.5, peri = 1 AU  =>  apo = 3 AU
        let apo = OrbitalCalculator::apoapsis(AU, 0.5).expect("elliptical has apoapsis");
        assert!((apo - 3.0 * AU).abs() / AU < 1e-9);
    }

    #[test]
    fn test_apoapsis_hyperbolic_is_none() {
        assert!(OrbitalCalculator::apoapsis(AU, 1.0).is_none());
        assert!(OrbitalCalculator::apoapsis(AU, 1.5).is_none());
    }

    #[test]
    fn test_periapsis_from_apoapsis() {
        let peri = OrbitalCalculator::periapsis(3.0 * AU, 0.5);
        assert!((peri - AU).abs() / AU < 1e-9);
    }

    #[test]
    fn test_semi_major_axis_from_apo_peri() {
        let a = OrbitalCalculator::semi_major_axis(3.0 * AU, AU);
        assert!((a - 2.0 * AU).abs() / AU < 1e-9);
    }

    // ── NBodySimulator: construction ──────────────────────────────────────────

    #[test]
    fn test_new_simulator_is_empty() {
        let sim = NBodySimulator::new();
        assert_eq!(sim.body_count(), 0);
    }

    #[test]
    fn test_add_body() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "Sun".to_string(),
            mass: SOLAR_MASS,
            position: [0.0; 3],
            velocity: [0.0; 3],
        });
        assert_eq!(sim.body_count(), 1);
    }

    #[test]
    fn test_body_lookup_found() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "Mars".to_string(),
            mass: 6.39e23,
            position: [0.0; 3],
            velocity: [0.0; 3],
        });
        assert!(sim.body("Mars").is_some());
        assert_eq!(sim.body("Mars").expect("body").name, "Mars");
    }

    #[test]
    fn test_body_lookup_not_found() {
        let sim = NBodySimulator::new();
        assert!(sim.body("Venus").is_none());
    }

    // ── NBodySimulator: step / step_n ─────────────────────────────────────────

    #[test]
    fn test_single_body_no_force() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "Loner".to_string(),
            mass: 1e20,
            position: [1e9, 0.0, 0.0],
            velocity: [1000.0, 0.0, 0.0],
        });
        let dt = 10.0;
        sim.step(dt);
        // With no other bodies, velocity should be unchanged and position updates by v*dt.
        let b = sim.body("Loner").expect("body");
        assert!((b.velocity[0] - 1000.0).abs() < 1e-9);
        assert!((b.position[0] - (1e9 + 1000.0 * dt)).abs() < 1e-9);
    }

    #[test]
    fn test_two_body_mutual_attraction() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "A".to_string(),
            mass: 1e20,
            position: [0.0, 0.0, 0.0],
            velocity: [0.0; 3],
        });
        sim.add_body(Body {
            name: "B".to_string(),
            mass: 1e20,
            position: [1e9, 0.0, 0.0],
            velocity: [0.0; 3],
        });
        sim.step(1.0);
        // A should have accelerated towards B (+x direction)
        let a = sim.body("A").expect("A");
        let b = sim.body("B").expect("B");
        assert!(a.velocity[0] > 0.0, "A should accelerate towards B");
        assert!(b.velocity[0] < 0.0, "B should accelerate towards A");
    }

    #[test]
    fn test_step_n_equivalent_to_n_steps() {
        let make_sim = || {
            let mut s = NBodySimulator::new();
            s.add_body(Body {
                name: "Star".to_string(),
                mass: SOLAR_MASS,
                position: [0.0; 3],
                velocity: [0.0; 3],
            });
            s.add_body(Body {
                name: "Planet".to_string(),
                mass: EARTH_MASS,
                position: [AU, 0.0, 0.0],
                velocity: [0.0, EARTH_ORBITAL_V, 0.0],
            });
            s
        };

        let mut sim_step = make_sim();
        let mut sim_step_n = make_sim();

        let dt = 100.0;
        let n = 20;

        for _ in 0..n {
            sim_step.step(dt);
        }
        sim_step_n.step_n(dt, n);

        let b1 = sim_step.body("Planet").expect("p");
        let b2 = sim_step_n.body("Planet").expect("p");
        assert!((b1.position[0] - b2.position[0]).abs() < 1e-3);
        assert!((b1.velocity[1] - b2.velocity[1]).abs() < 1e-3);
    }

    // ── NBodySimulator: energy ────────────────────────────────────────────────

    #[test]
    fn test_kinetic_energy_single_body() {
        let mut sim = NBodySimulator::new();
        // KE = 0.5 * 2 * 3^2 = 9
        sim.add_body(Body {
            name: "A".to_string(),
            mass: 2.0,
            position: [0.0; 3],
            velocity: [3.0, 0.0, 0.0],
        });
        assert!((sim.kinetic_energy() - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_kinetic_energy_zero_velocity() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "A".to_string(),
            mass: 1e20,
            position: [0.0; 3],
            velocity: [0.0; 3],
        });
        assert_eq!(sim.kinetic_energy(), 0.0);
    }

    #[test]
    fn test_potential_energy_two_bodies() {
        let mut sim = NBodySimulator::new();
        let m1 = 1e20;
        let m2 = 1e20;
        let r = 1e9;
        sim.add_body(Body {
            name: "A".to_string(),
            mass: m1,
            position: [0.0, 0.0, 0.0],
            velocity: [0.0; 3],
        });
        sim.add_body(Body {
            name: "B".to_string(),
            mass: m2,
            position: [r, 0.0, 0.0],
            velocity: [0.0; 3],
        });
        let expected = -G * m1 * m2 / r;
        assert!((sim.potential_energy() - expected).abs() / expected.abs() < 1e-9);
    }

    #[test]
    fn test_potential_energy_single_body_zero() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "A".to_string(),
            mass: 1e20,
            position: [0.0; 3],
            velocity: [0.0; 3],
        });
        assert_eq!(sim.potential_energy(), 0.0);
    }

    #[test]
    fn test_total_energy_is_sum() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "A".to_string(),
            mass: 2.0,
            position: [0.0, 0.0, 0.0],
            velocity: [3.0, 0.0, 0.0],
        });
        sim.add_body(Body {
            name: "B".to_string(),
            mass: 2.0,
            position: [1e9, 0.0, 0.0],
            velocity: [0.0; 3],
        });
        let ke = sim.kinetic_energy();
        let pe = sim.potential_energy();
        assert!((sim.total_energy() - (ke + pe)).abs() < 1e-9);
    }

    #[test]
    fn test_energy_roughly_conserved_short_run() {
        // For a short simulation with a small time step the total energy should
        // not drift by more than 1 % (Euler has first-order error).
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "Sun".to_string(),
            mass: SOLAR_MASS,
            position: [0.0; 3],
            velocity: [0.0; 3],
        });
        sim.add_body(Body {
            name: "Earth".to_string(),
            mass: EARTH_MASS,
            position: [AU, 0.0, 0.0],
            velocity: [0.0, EARTH_ORBITAL_V, 0.0],
        });
        let e0 = sim.total_energy();
        sim.step_n(60.0, 10); // 10 minutes total, tiny step
        let e1 = sim.total_energy();
        let rel_change = (e1 - e0).abs() / e0.abs();
        assert!(rel_change < 0.01, "energy drift {rel_change:.2e} too large");
    }

    // ── NBodySimulator: centre_of_mass / momentum ─────────────────────────────

    #[test]
    fn test_centre_of_mass_two_equal_masses() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "A".to_string(),
            mass: 1.0,
            position: [-1.0, 0.0, 0.0],
            velocity: [0.0; 3],
        });
        sim.add_body(Body {
            name: "B".to_string(),
            mass: 1.0,
            position: [1.0, 0.0, 0.0],
            velocity: [0.0; 3],
        });
        let com = sim.centre_of_mass();
        assert!((com[0]).abs() < 1e-9);
    }

    #[test]
    fn test_total_momentum_two_bodies() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "A".to_string(),
            mass: 2.0,
            position: [0.0; 3],
            velocity: [3.0, 0.0, 0.0],
        });
        sim.add_body(Body {
            name: "B".to_string(),
            mass: 2.0,
            position: [1.0, 0.0, 0.0],
            velocity: [-3.0, 0.0, 0.0],
        });
        let p = sim.total_momentum();
        assert!(p[0].abs() < 1e-9); // net x-momentum = 0
    }

    // ── Body helpers ──────────────────────────────────────────────────────────

    #[test]
    fn test_body_kinetic_energy() {
        let b = Body {
            name: "t".to_string(),
            mass: 2.0,
            position: [0.0; 3],
            velocity: [3.0, 4.0, 0.0],
        };
        // KE = 0.5 * 2 * (9 + 16) = 25
        assert!((b.kinetic_energy() - 25.0).abs() < 1e-9);
    }

    // ── OrbitalCalculator: specific energy / hill sphere ─────────────────────

    #[test]
    fn test_specific_orbital_energy_negative() {
        let e = OrbitalCalculator::specific_orbital_energy(SOLAR_MASS, AU);
        assert!(e < 0.0, "bound orbit must have negative energy");
    }

    #[test]
    fn test_hill_sphere_earth_in_solar_orbit() {
        let r_h = OrbitalCalculator::hill_sphere(AU, EARTH_MASS, SOLAR_MASS);
        // Earth Hill sphere ~1.5 million km = 1.5e9 m
        assert!(
            r_h > 1e9 && r_h < 5e9,
            "Hill sphere radius {r_h:.2e} m out of expected range"
        );
    }

    // ── default impl ──────────────────────────────────────────────────────────

    #[test]
    fn test_default_simulator() {
        let sim = NBodySimulator::default();
        assert_eq!(sim.body_count(), 0);
    }

    // ── bodies accessor ───────────────────────────────────────────────────────

    #[test]
    fn test_bodies_accessor() {
        let mut sim = NBodySimulator::new();
        sim.add_body(Body {
            name: "X".to_string(),
            mass: 1e10,
            position: [0.0; 3],
            velocity: [0.0; 3],
        });
        assert_eq!(sim.bodies().len(), 1);
        assert_eq!(sim.bodies()[0].name, "X");
    }

    // ── G constant ────────────────────────────────────────────────────────────

    #[test]
    fn test_g_constant_value() {
        assert!((G - 6.674e-11).abs() < 1e-15);
    }
}
