//! SWRL (Semantic Web Rule Language) - Statistics
//!
//! This module implements SWRL rule components.


pub struct SwrlStats {
    pub total_rules: usize,
    pub total_builtins: usize,
}

impl std::fmt::Display for SwrlStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rules: {}, Built-ins: {}",
            self.total_rules, self.total_builtins
        )
    }
}
