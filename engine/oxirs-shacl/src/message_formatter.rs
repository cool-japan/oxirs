// SHACL sh:message template interpolation
// Added in v1.1.0 Round 7

use std::collections::HashMap;

/// A parsed token in a message template.
#[derive(Debug, Clone, PartialEq)]
pub enum MessageToken {
    Text(String),
    Placeholder(String), // {varName}
}

/// A parsed SHACL message template.
#[derive(Debug, Clone)]
pub struct MessageTemplate {
    pub tokens: Vec<MessageToken>,
    pub lang: Option<String>,
}

/// A formatted validation message.
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    pub text: String,
    pub lang: Option<String>,
    pub severity: MessageSeverity,
}

/// Severity of a SHACL validation message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MessageSeverity {
    Violation,
    Warning,
    Info,
}

impl std::fmt::Display for MessageSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageSeverity::Violation => write!(f, "Violation"),
            MessageSeverity::Warning => write!(f, "Warning"),
            MessageSeverity::Info => write!(f, "Info"),
        }
    }
}

/// Errors that can occur during message formatting.
#[derive(Debug)]
pub enum MessageError {
    UnknownPlaceholder(String),
    TemplateParseFailed(String),
}

impl std::fmt::Display for MessageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageError::UnknownPlaceholder(name) => {
                write!(f, "Unknown placeholder: {{{name}}}")
            }
            MessageError::TemplateParseFailed(reason) => {
                write!(f, "Template parse failed: {reason}")
            }
        }
    }
}

impl std::error::Error for MessageError {}

/// Formatter for SHACL sh:message templates.
pub struct MessageFormatter;

impl MessageFormatter {
    /// Parse a template string into tokens.
    ///
    /// Template syntax: `{varName}` for placeholders, all other text is literal.
    pub fn parse_template(template: &str) -> Result<MessageTemplate, MessageError> {
        let mut tokens = Vec::new();
        let mut current_text = String::new();
        let mut chars = template.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                // Flush pending text
                if !current_text.is_empty() {
                    tokens.push(MessageToken::Text(current_text.clone()));
                    current_text.clear();
                }
                // Read placeholder name until '}'
                let mut name = String::new();
                let mut closed = false;
                for inner in chars.by_ref() {
                    if inner == '}' {
                        closed = true;
                        break;
                    }
                    name.push(inner);
                }
                if !closed {
                    return Err(MessageError::TemplateParseFailed(
                        "unclosed placeholder '{' without matching '}'".to_string(),
                    ));
                }
                if name.is_empty() {
                    return Err(MessageError::TemplateParseFailed(
                        "empty placeholder '{}'".to_string(),
                    ));
                }
                tokens.push(MessageToken::Placeholder(name));
            } else {
                current_text.push(ch);
            }
        }
        if !current_text.is_empty() {
            tokens.push(MessageToken::Text(current_text));
        }
        Ok(MessageTemplate { tokens, lang: None })
    }

    /// Parse a template with an optional language tag.
    pub fn parse_template_with_lang(
        template: &str,
        lang: Option<String>,
    ) -> Result<MessageTemplate, MessageError> {
        let mut tpl = Self::parse_template(template)?;
        tpl.lang = lang;
        Ok(tpl)
    }

    /// Interpolate a template with variable bindings.
    ///
    /// Returns an error if any placeholder has no matching binding.
    pub fn format(
        template: &MessageTemplate,
        bindings: &HashMap<String, String>,
        severity: MessageSeverity,
    ) -> Result<ValidationMessage, MessageError> {
        let mut text = String::new();
        for token in &template.tokens {
            match token {
                MessageToken::Text(s) => text.push_str(s),
                MessageToken::Placeholder(name) => match bindings.get(name) {
                    Some(value) => text.push_str(value),
                    None => {
                        return Err(MessageError::UnknownPlaceholder(name.clone()));
                    }
                },
            }
        }
        Ok(ValidationMessage {
            text,
            lang: template.lang.clone(),
            severity,
        })
    }

    /// Build a standard SHACL violation message.
    pub fn violation(
        constraint_name: &str,
        focus_node: &str,
        value: Option<&str>,
        message: Option<&str>,
    ) -> ValidationMessage {
        let text = if let Some(msg) = message {
            msg.to_string()
        } else if let Some(val) = value {
            format!(
                "Constraint violation [{constraint_name}]: focus node <{focus_node}>, value: {val}"
            )
        } else {
            format!("Constraint violation [{constraint_name}]: focus node <{focus_node}>")
        };
        ValidationMessage {
            text,
            lang: None,
            severity: MessageSeverity::Violation,
        }
    }

    /// Format multiple messages into a report string.
    pub fn format_report(messages: &[ValidationMessage]) -> String {
        let mut out = String::new();
        out.push_str("=== SHACL Validation Report ===\n");
        if messages.is_empty() {
            out.push_str("No violations found.\n");
        } else {
            for (i, msg) in messages.iter().enumerate() {
                let lang_part = msg
                    .lang
                    .as_deref()
                    .map(|l| format!(" [{l}]"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "{}. [{}]{} {}\n",
                    i + 1,
                    msg.severity,
                    lang_part,
                    msg.text
                ));
            }
        }
        out
    }

    /// Parse a SHACL severity IRI into a `MessageSeverity`.
    ///
    /// Recognises the three standard SHACL severity IRIs.
    /// Unknown IRIs default to `Violation`.
    pub fn parse_severity(iri: &str) -> MessageSeverity {
        match iri {
            "sh:Violation" | "http://www.w3.org/ns/shacl#Violation" => MessageSeverity::Violation,
            "sh:Warning" | "http://www.w3.org/ns/shacl#Warning" => MessageSeverity::Warning,
            "sh:Info" | "http://www.w3.org/ns/shacl#Info" => MessageSeverity::Info,
            _ => MessageSeverity::Violation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bindings(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ---- parse_template ----

    #[test]
    fn test_parse_no_placeholders() {
        let tpl = MessageFormatter::parse_template("Hello, world!").unwrap();
        assert_eq!(tpl.tokens.len(), 1);
        assert_eq!(
            tpl.tokens[0],
            MessageToken::Text("Hello, world!".to_string())
        );
    }

    #[test]
    fn test_parse_empty_template() {
        let tpl = MessageFormatter::parse_template("").unwrap();
        assert!(tpl.tokens.is_empty());
    }

    #[test]
    fn test_parse_single_placeholder() {
        let tpl = MessageFormatter::parse_template("{name}").unwrap();
        assert_eq!(tpl.tokens.len(), 1);
        assert_eq!(tpl.tokens[0], MessageToken::Placeholder("name".to_string()));
    }

    #[test]
    fn test_parse_placeholder_with_surrounding_text() {
        let tpl = MessageFormatter::parse_template("Value {val} is invalid").unwrap();
        assert_eq!(tpl.tokens.len(), 3);
        assert_eq!(tpl.tokens[0], MessageToken::Text("Value ".to_string()));
        assert_eq!(tpl.tokens[1], MessageToken::Placeholder("val".to_string()));
        assert_eq!(tpl.tokens[2], MessageToken::Text(" is invalid".to_string()));
    }

    #[test]
    fn test_parse_multiple_placeholders() {
        let tpl = MessageFormatter::parse_template("{a} and {b}").unwrap();
        let placeholders: Vec<_> = tpl
            .tokens
            .iter()
            .filter(|t| matches!(t, MessageToken::Placeholder(_)))
            .collect();
        assert_eq!(placeholders.len(), 2);
    }

    #[test]
    fn test_parse_adjacent_placeholders() {
        let tpl = MessageFormatter::parse_template("{x}{y}").unwrap();
        assert_eq!(tpl.tokens.len(), 2);
        assert_eq!(tpl.tokens[0], MessageToken::Placeholder("x".to_string()));
        assert_eq!(tpl.tokens[1], MessageToken::Placeholder("y".to_string()));
    }

    #[test]
    fn test_parse_unclosed_brace_error() {
        let result = MessageFormatter::parse_template("Hello {unclosed");
        assert!(matches!(result, Err(MessageError::TemplateParseFailed(_))));
    }

    #[test]
    fn test_parse_empty_placeholder_error() {
        let result = MessageFormatter::parse_template("{}");
        assert!(matches!(result, Err(MessageError::TemplateParseFailed(_))));
    }

    #[test]
    fn test_parse_template_with_lang() {
        let tpl = MessageFormatter::parse_template_with_lang(
            "{value} is invalid",
            Some("en".to_string()),
        )
        .unwrap();
        assert_eq!(tpl.lang, Some("en".to_string()));
        assert!(!tpl.tokens.is_empty());
    }

    // ---- format ----

    #[test]
    fn test_format_all_placeholders_filled() {
        let tpl = MessageFormatter::parse_template("{name} must be {type}").unwrap();
        let b = bindings(&[("name", "Alice"), ("type", "Person")]);
        let msg = MessageFormatter::format(&tpl, &b, MessageSeverity::Violation).unwrap();
        assert_eq!(msg.text, "Alice must be Person");
        assert_eq!(msg.severity, MessageSeverity::Violation);
    }

    #[test]
    fn test_format_unknown_placeholder_error() {
        let tpl = MessageFormatter::parse_template("{name} is {age}").unwrap();
        let b = bindings(&[("name", "Bob")]);
        let result = MessageFormatter::format(&tpl, &b, MessageSeverity::Warning);
        assert!(matches!(result, Err(MessageError::UnknownPlaceholder(_))));
    }

    #[test]
    fn test_format_no_placeholders() {
        let tpl = MessageFormatter::parse_template("Static message").unwrap();
        let b = bindings(&[]);
        let msg = MessageFormatter::format(&tpl, &b, MessageSeverity::Info).unwrap();
        assert_eq!(msg.text, "Static message");
        assert_eq!(msg.severity, MessageSeverity::Info);
    }

    #[test]
    fn test_format_severity_warning() {
        let tpl = MessageFormatter::parse_template("warning").unwrap();
        let b = bindings(&[]);
        let msg = MessageFormatter::format(&tpl, &b, MessageSeverity::Warning).unwrap();
        assert_eq!(msg.severity, MessageSeverity::Warning);
    }

    #[test]
    fn test_format_severity_info() {
        let tpl = MessageFormatter::parse_template("info").unwrap();
        let b = bindings(&[]);
        let msg = MessageFormatter::format(&tpl, &b, MessageSeverity::Info).unwrap();
        assert_eq!(msg.severity, MessageSeverity::Info);
    }

    #[test]
    fn test_format_with_lang() {
        let tpl =
            MessageFormatter::parse_template_with_lang("{x}", Some("de".to_string())).unwrap();
        let b = bindings(&[("x", "Wert")]);
        let msg = MessageFormatter::format(&tpl, &b, MessageSeverity::Violation).unwrap();
        assert_eq!(msg.lang, Some("de".to_string()));
        assert_eq!(msg.text, "Wert");
    }

    // ---- violation builder ----

    #[test]
    fn test_violation_with_message() {
        let msg =
            MessageFormatter::violation("minCount", "http://ex/s", None, Some("Too few values"));
        assert_eq!(msg.text, "Too few values");
        assert_eq!(msg.severity, MessageSeverity::Violation);
    }

    #[test]
    fn test_violation_without_message_with_value() {
        let msg =
            MessageFormatter::violation("maxLength", "http://ex/s", Some("toolongstring"), None);
        assert!(msg.text.contains("maxLength"));
        assert!(msg.text.contains("http://ex/s"));
        assert!(msg.text.contains("toolongstring"));
    }

    #[test]
    fn test_violation_without_message_without_value() {
        let msg = MessageFormatter::violation("type", "http://ex/s", None, None);
        assert!(msg.text.contains("type"));
        assert!(msg.text.contains("http://ex/s"));
        assert_eq!(msg.severity, MessageSeverity::Violation);
    }

    #[test]
    fn test_violation_lang_is_none() {
        let msg = MessageFormatter::violation("c", "n", None, None);
        assert!(msg.lang.is_none());
    }

    // ---- format_report ----

    #[test]
    fn test_format_report_empty() {
        let report = MessageFormatter::format_report(&[]);
        assert!(report.contains("No violations"));
    }

    #[test]
    fn test_format_report_with_messages() {
        let msgs = vec![
            ValidationMessage {
                text: "First violation".to_string(),
                lang: None,
                severity: MessageSeverity::Violation,
            },
            ValidationMessage {
                text: "A warning".to_string(),
                lang: None,
                severity: MessageSeverity::Warning,
            },
        ];
        let report = MessageFormatter::format_report(&msgs);
        assert!(report.contains("First violation"));
        assert!(report.contains("A warning"));
        assert!(report.contains("1."));
        assert!(report.contains("2."));
    }

    #[test]
    fn test_format_report_shows_severity() {
        let msgs = vec![ValidationMessage {
            text: "x".to_string(),
            lang: None,
            severity: MessageSeverity::Info,
        }];
        let report = MessageFormatter::format_report(&msgs);
        assert!(report.contains("Info"));
    }

    #[test]
    fn test_format_report_shows_lang() {
        let msgs = vec![ValidationMessage {
            text: "x".to_string(),
            lang: Some("fr".to_string()),
            severity: MessageSeverity::Violation,
        }];
        let report = MessageFormatter::format_report(&msgs);
        assert!(report.contains("fr"));
    }

    // ---- parse_severity ----

    #[test]
    fn test_parse_severity_violation_short() {
        assert_eq!(
            MessageFormatter::parse_severity("sh:Violation"),
            MessageSeverity::Violation
        );
    }

    #[test]
    fn test_parse_severity_violation_full() {
        assert_eq!(
            MessageFormatter::parse_severity("http://www.w3.org/ns/shacl#Violation"),
            MessageSeverity::Violation
        );
    }

    #[test]
    fn test_parse_severity_warning_short() {
        assert_eq!(
            MessageFormatter::parse_severity("sh:Warning"),
            MessageSeverity::Warning
        );
    }

    #[test]
    fn test_parse_severity_warning_full() {
        assert_eq!(
            MessageFormatter::parse_severity("http://www.w3.org/ns/shacl#Warning"),
            MessageSeverity::Warning
        );
    }

    #[test]
    fn test_parse_severity_info_short() {
        assert_eq!(
            MessageFormatter::parse_severity("sh:Info"),
            MessageSeverity::Info
        );
    }

    #[test]
    fn test_parse_severity_info_full() {
        assert_eq!(
            MessageFormatter::parse_severity("http://www.w3.org/ns/shacl#Info"),
            MessageSeverity::Info
        );
    }

    #[test]
    fn test_parse_severity_unknown_defaults_violation() {
        assert_eq!(
            MessageFormatter::parse_severity("sh:UnknownSeverity"),
            MessageSeverity::Violation
        );
    }

    #[test]
    fn test_parse_severity_empty_defaults_violation() {
        assert_eq!(
            MessageFormatter::parse_severity(""),
            MessageSeverity::Violation
        );
    }

    // ---- MessageError display ----

    #[test]
    fn test_message_error_unknown_placeholder_display() {
        let err = MessageError::UnknownPlaceholder("myVar".to_string());
        let s = format!("{err}");
        assert!(s.contains("myVar"));
    }

    #[test]
    fn test_message_error_template_parse_failed_display() {
        let err = MessageError::TemplateParseFailed("unclosed".to_string());
        let s = format!("{err}");
        assert!(s.contains("unclosed"));
    }

    // ---- MessageSeverity display ----

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", MessageSeverity::Violation), "Violation");
        assert_eq!(format!("{}", MessageSeverity::Warning), "Warning");
        assert_eq!(format!("{}", MessageSeverity::Info), "Info");
    }
}
