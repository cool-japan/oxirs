//! Prompt construction helpers for SHACL-specific use cases.
//!
//! [`ShaclPrompts`] provides factory methods that return ready-to-use
//! [`Vec<Message>`](super::provider::Message) lists suitable for passing
//! directly to [`CompletionProvider::complete`](super::provider::CompletionProvider::complete).

use super::provider::{Message, Role};

/// Collection of SHACL-specific prompt builders.
pub struct ShaclPrompts;

impl ShaclPrompts {
    /// Build a system + user prompt for generating SHACL shapes from a
    /// natural-language constraint description.
    ///
    /// The expected LLM output format is:
    ///
    /// ```json
    /// {
    ///   "node_shape": {
    ///     "target_class": "<URI or prefixed name>",
    ///     "properties": [
    ///       {
    ///         "path": "<URI or prefixed name>",
    ///         "min_count": 1,
    ///         "max_count": null,
    ///         "datatype": "<URI or null>"
    ///       }
    ///     ]
    ///   }
    /// }
    /// ```
    pub fn shape_generation_prompt(constraint_text: &str) -> Vec<Message> {
        vec![
            Message {
                role: Role::System,
                content: concat!(
                    "You are a SHACL expert. Generate a SHACL shape from the user's natural ",
                    "language description. Reply with valid JSON matching this schema exactly:\n",
                    r#"{"node_shape": {"target_class": "...", "properties": [{"path": "...", "#,
                    r#""min_count": 1, "max_count": null, "datatype": "..."}]}}"#,
                    "\nDo not include any other text or explanation."
                )
                .to_string(),
            },
            Message {
                role: Role::User,
                content: constraint_text.to_string(),
            },
        ]
    }

    /// Build a prompt for explaining a SHACL constraint violation in plain English.
    ///
    /// The returned explanation is intended for end-users who may not be
    /// familiar with SHACL terminology.
    pub fn violation_explanation_prompt(violation_summary: &str) -> Vec<Message> {
        vec![
            Message {
                role: Role::System,
                content: concat!(
                    "You are a SHACL validator assistant. Explain the following SHACL constraint ",
                    "violation in plain English for a non-technical user. Be concise and helpful.",
                )
                .to_string(),
            },
            Message {
                role: Role::User,
                content: violation_summary.to_string(),
            },
        ]
    }

    /// Build a prompt for suggesting fixes for a SHACL violation.
    pub fn fix_suggestion_prompt(violation_summary: &str) -> Vec<Message> {
        vec![
            Message {
                role: Role::System,
                content: concat!(
                    "You are a SHACL validator assistant. Given the following SHACL constraint ",
                    "violation, suggest concrete steps the data owner can take to fix the ",
                    "violation. Be specific and actionable.",
                )
                .to_string(),
            },
            Message {
                role: Role::User,
                content: violation_summary.to_string(),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_generation_prompt_has_both_roles() {
        let msgs = ShaclPrompts::shape_generation_prompt("Every person must have a name.");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].role, Role::User);
        assert!(msgs[1].content.contains("person"));
    }

    #[test]
    fn test_violation_explanation_prompt_contains_violation() {
        let summary = "sh:minCount violation on foaf:name";
        let msgs = ShaclPrompts::violation_explanation_prompt(summary);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[1].content.contains("foaf:name"));
    }

    #[test]
    fn test_fix_suggestion_prompt_non_empty() {
        let msgs = ShaclPrompts::fix_suggestion_prompt("Missing required property ex:id");
        assert_eq!(msgs.len(), 2);
        assert!(!msgs[0].content.is_empty());
        assert!(!msgs[1].content.is_empty());
    }
}
