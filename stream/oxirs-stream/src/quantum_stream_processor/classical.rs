//! Classical processing components

/// Classical processor for hybrid quantum-classical algorithms
pub struct ClassicalProcessor;

impl ClassicalProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ClassicalProcessor {
    fn default() -> Self {
        Self::new()
    }
}