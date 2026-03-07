//! # StreamingExecutor - create_optimized_bgp_join_stream_group Methods
//!
//! This module contains method implementations for `StreamingExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::TriplePattern;
use anyhow::Result;

use super::functions::DataStream;

use super::streamingexecutor_type::StreamingExecutor;

impl StreamingExecutor {
    /// Create optimized BGP join stream
    pub(super) fn create_optimized_bgp_join_stream(
        &mut self,
        patterns: &[TriplePattern],
    ) -> Result<Box<dyn DataStream>> {
        let mut ordered_patterns = patterns.to_vec();
        self.order_patterns_for_streaming(&mut ordered_patterns);
        let mut result_stream = self.create_pattern_stream(&ordered_patterns[0])?;
        for pattern in &ordered_patterns[1..] {
            let pattern_stream = self.create_pattern_stream(pattern)?;
            let join_variables = self.find_join_variables_between_streams(
                result_stream.as_ref(),
                pattern_stream.as_ref(),
            )?;
            result_stream =
                self.create_streaming_join(result_stream, pattern_stream, join_variables)?;
        }
        Ok(result_stream)
    }
    /// Order patterns for optimal streaming execution
    pub(super) fn order_patterns_for_streaming(&self, patterns: &mut [TriplePattern]) {
        patterns.sort_by(|a, b| {
            let card_a = self.estimate_pattern_cardinality(a);
            let card_b = self.estimate_pattern_cardinality(b);
            card_a.cmp(&card_b)
        });
    }
}
