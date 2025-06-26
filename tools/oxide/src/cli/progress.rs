//! Progress tracking for long-running operations
//!
//! Provides various progress indicators and tracking mechanisms.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::sync::Arc;
use std::time::Duration;

/// Progress tracker for managing multiple progress bars
pub struct ProgressTracker {
    multi: Arc<MultiProgress>,
    bars: Vec<ProgressBar>,
}

/// Type of progress indicator
#[derive(Debug, Clone)]
pub enum ProgressType {
    /// Simple spinner for indeterminate progress
    Spinner,
    /// Progress bar with known total
    Bar(u64),
    /// Bytes progress (shows transfer rate)
    Bytes(u64),
    /// Counter (just shows count)
    Counter,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new() -> Self {
        Self {
            multi: Arc::new(MultiProgress::new()),
            bars: Vec::new(),
        }
    }

    /// Create a progress tracker that outputs to stderr
    pub fn new_stderr() -> Self {
        let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
        Self {
            multi: Arc::new(multi),
            bars: Vec::new(),
        }
    }

    /// Add a new progress indicator
    pub fn add_progress(&mut self, name: &str, progress_type: ProgressType) -> ProgressBar {
        let pb = match progress_type {
            ProgressType::Spinner => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} {prefix:>12.cyan.dim} {msg}")
                        .unwrap()
                        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
                );
                pb.enable_steady_tick(Duration::from_millis(100));
                pb
            }
            ProgressType::Bar(total) => {
                let pb = ProgressBar::new(total);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} {prefix:>12.cyan.dim} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                        .unwrap()
                        .progress_chars("=>-"),
                );
                pb
            }
            ProgressType::Bytes(total) => {
                let pb = ProgressBar::new(total);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} {prefix:>12.cyan.dim} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}) {msg}")
                        .unwrap()
                        .progress_chars("=>-"),
                );
                pb
            }
            ProgressType::Counter => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} {prefix:>12.cyan.dim} {pos} {msg}")
                        .unwrap(),
                );
                pb.enable_steady_tick(Duration::from_millis(100));
                pb
            }
        };

        pb.set_prefix(name.to_string());
        let pb = self.multi.add(pb);
        self.bars.push(pb.clone());
        pb
    }

    /// Create a sub-progress bar under a parent
    pub fn add_sub_progress(
        &mut self,
        parent: &ProgressBar,
        name: &str,
        progress_type: ProgressType,
    ) -> ProgressBar {
        let pb = self.add_progress(name, progress_type);
        self.multi.insert_after(parent, pb.clone());
        pb
    }

    /// Clear all progress bars
    pub fn clear(&self) {
        self.multi.clear().ok();
    }

    /// Finish all progress bars
    pub fn finish_all(&mut self) {
        for pb in &self.bars {
            pb.finish_and_clear();
        }
        self.bars.clear();
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Progress builder for fluent API
pub struct ProgressBuilder {
    message: String,
    total: Option<u64>,
    style: ProgressStyle,
}

impl ProgressBuilder {
    /// Create a new progress builder
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            total: None,
            style: ProgressStyle::default_bar(),
        }
    }

    /// Set the total for the progress bar
    pub fn with_total(mut self, total: u64) -> Self {
        self.total = Some(total);
        self
    }

    /// Use a custom style
    pub fn with_style(mut self, style: ProgressStyle) -> Self {
        self.style = style;
        self
    }

    /// Use bytes style (shows transfer rate)
    pub fn bytes_style(mut self) -> Self {
        self.style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}) {msg}")
            .unwrap()
            .progress_chars("=>-");
        self
    }

    /// Use spinner style
    pub fn spinner_style(mut self) -> Self {
        self.style = ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap();
        self
    }

    /// Build the progress bar
    pub fn build(self) -> ProgressBar {
        let pb = match self.total {
            Some(total) => ProgressBar::new(total),
            None => ProgressBar::new_spinner(),
        };

        pb.set_style(self.style);
        pb.set_message(self.message);

        if self.total.is_none() {
            pb.enable_steady_tick(Duration::from_millis(100));
        }

        pb
    }
}

/// Helper functions for common progress patterns
pub mod helpers {
    use super::*;
    use std::path::Path;

    /// Create a file processing progress bar
    pub fn file_progress(file_count: u64) -> ProgressBar {
        ProgressBuilder::new("Processing files")
            .with_total(file_count)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} {msg:30} [{bar:40.cyan/blue}] {pos}/{len} files")
                    .unwrap()
                    .progress_chars("=>-"),
            )
            .build()
    }

    /// Create a download progress bar
    pub fn download_progress(total_bytes: u64, url: &str) -> ProgressBar {
        let filename = Path::new(url).file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");

        ProgressBuilder::new(format!("Downloading {}", filename))
            .with_total(total_bytes)
            .bytes_style()
            .build()
    }

    /// Create a query execution progress spinner
    pub fn query_progress() -> ProgressBar {
        ProgressBuilder::new("Executing query")
            .spinner_style()
            .build()
    }

    /// Create a validation progress bar
    pub fn validation_progress(total_items: u64) -> ProgressBar {
        ProgressBuilder::new("Validating")
            .with_total(total_items)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} {msg:20} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
                    .unwrap()
                    .progress_chars("=>-"),
            )
            .build()
    }

    /// Create a multi-file import progress
    pub fn import_progress(file: &str, current: usize, total: usize) -> ProgressBar {
        ProgressBuilder::new(format!("Importing ({}/{}) {}", current, total, file))
            .spinner_style()
            .build()
    }
}

/// Progress callback for operations that report progress
pub trait ProgressCallback: Send + Sync {
    /// Update progress
    fn update(&self, current: u64, total: Option<u64>, message: Option<&str>);
    
    /// Mark as completed
    fn finish(&self, message: Option<&str>);
    
    /// Report an error
    fn error(&self, message: &str);
}

/// Implementation of ProgressCallback using a ProgressBar
impl ProgressCallback for ProgressBar {
    fn update(&self, current: u64, total: Option<u64>, message: Option<&str>) {
        self.set_position(current);
        if let Some(total) = total {
            self.set_length(total);
        }
        if let Some(msg) = message {
            self.set_message(msg.to_string());
        }
    }

    fn finish(&self, message: Option<&str>) {
        if let Some(msg) = message {
            self.finish_with_message(msg.to_string());
        } else {
            self.finish();
        }
    }

    fn error(&self, message: &str) {
        self.abandon_with_message(format!("Error: {}", message));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_builder() {
        let pb = ProgressBuilder::new("Testing")
            .with_total(100)
            .build();
        
        assert_eq!(pb.length().unwrap(), 100);
        pb.finish_and_clear();
    }

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::new();
        
        let pb1 = tracker.add_progress("Task 1", ProgressType::Bar(100));
        let pb2 = tracker.add_progress("Task 2", ProgressType::Spinner);
        
        pb1.inc(50);
        pb2.tick();
        
        tracker.finish_all();
    }

    #[test]
    fn test_progress_helpers() {
        let pb = helpers::file_progress(10);
        pb.inc(5);
        assert_eq!(pb.position(), 5);
        pb.finish_and_clear();

        let pb = helpers::query_progress();
        pb.tick();
        pb.finish_and_clear();
    }
}