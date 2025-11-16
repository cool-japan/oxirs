//! Export Module
//!
//! Provides functionality to export chat conversations and query results
//! to multiple formats (JSON, CSV, XML, Markdown).

use crate::messages::Message;
use crate::session_manager::SessionData;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info};

/// Export format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// XML format
    XML,
    /// Markdown format
    Markdown,
    /// HTML format
    HTML,
    /// Plain text format
    Text,
}

impl ExportFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &str {
        match self {
            ExportFormat::JSON => "json",
            ExportFormat::CSV => "csv",
            ExportFormat::XML => "xml",
            ExportFormat::Markdown => "md",
            ExportFormat::HTML => "html",
            ExportFormat::Text => "txt",
        }
    }

    /// Get MIME type for this format
    pub fn mime_type(&self) -> &str {
        match self {
            ExportFormat::JSON => "application/json",
            ExportFormat::CSV => "text/csv",
            ExportFormat::XML => "application/xml",
            ExportFormat::Markdown => "text/markdown",
            ExportFormat::HTML => "text/html",
            ExportFormat::Text => "text/plain",
        }
    }
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Include metadata
    pub include_metadata: bool,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Include user IDs
    pub include_user_ids: bool,
    /// Include message IDs
    pub include_message_ids: bool,
    /// Pretty print (for JSON/XML)
    pub pretty_print: bool,
    /// Include session statistics
    pub include_statistics: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            include_metadata: true,
            include_timestamps: true,
            include_user_ids: true,
            include_message_ids: true,
            pretty_print: true,
            include_statistics: true,
        }
    }
}

/// Exporter for chat data
pub struct ChatExporter {
    config: ExportConfig,
}

impl ChatExporter {
    /// Create a new exporter
    pub fn new(config: ExportConfig) -> Self {
        info!("Initialized chat exporter");
        Self { config }
    }

    /// Export session to the specified format
    pub fn export_session(&self, session: &SessionData, format: ExportFormat) -> Result<String> {
        debug!("Exporting session {} to {:?} format", session.id, format);

        match format {
            ExportFormat::JSON => self.export_json(session),
            ExportFormat::CSV => self.export_csv(session),
            ExportFormat::XML => self.export_xml(session),
            ExportFormat::Markdown => self.export_markdown(session),
            ExportFormat::HTML => self.export_html(session),
            ExportFormat::Text => self.export_text(session),
        }
    }

    /// Export messages to the specified format
    pub fn export_messages(&self, messages: &[Message], format: ExportFormat) -> Result<String> {
        debug!(
            "Exporting {} messages to {:?} format",
            messages.len(),
            format
        );

        match format {
            ExportFormat::JSON => self.messages_to_json(messages),
            ExportFormat::CSV => self.messages_to_csv(messages),
            ExportFormat::XML => self.messages_to_xml(messages),
            ExportFormat::Markdown => self.messages_to_markdown(messages),
            ExportFormat::HTML => self.messages_to_html(messages),
            ExportFormat::Text => self.messages_to_text(messages),
        }
    }

    /// Export to JSON format
    fn export_json(&self, session: &SessionData) -> Result<String> {
        if self.config.pretty_print {
            serde_json::to_string_pretty(session).context("Failed to serialize to JSON")
        } else {
            serde_json::to_string(session).context("Failed to serialize to JSON")
        }
    }

    /// Export to CSV format
    fn export_csv(&self, session: &SessionData) -> Result<String> {
        let mut csv = String::new();

        // Header
        let mut headers = vec!["Role", "Content"];
        if self.config.include_timestamps {
            headers.push("Timestamp");
        }
        if self.config.include_message_ids {
            headers.push("Message ID");
        }
        csv.push_str(&headers.join(","));
        csv.push('\n');

        // Data rows
        for message in &session.messages {
            let mut row = vec![
                format!("{:?}", message.role),
                self.escape_csv(&message.content.to_string()),
            ];

            if self.config.include_timestamps {
                row.push(message.timestamp.to_rfc3339());
            }

            if self.config.include_message_ids {
                row.push(message.id.clone());
            }

            csv.push_str(&row.join(","));
            csv.push('\n');
        }

        Ok(csv)
    }

    /// Export to XML format
    fn export_xml(&self, session: &SessionData) -> Result<String> {
        let mut xml = String::new();

        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<session>\n");
        xml.push_str(&format!("  <id>{}</id>\n", session.id));
        xml.push_str(&format!(
            "  <created_at>{}</created_at>\n",
            session.created_at
        ));

        xml.push_str("  <messages>\n");
        for message in &session.messages {
            xml.push_str("    <message>\n");
            if self.config.include_message_ids {
                xml.push_str(&format!("      <id>{}</id>\n", message.id));
            }
            xml.push_str(&format!("      <role>{:?}</role>\n", message.role));
            xml.push_str(&format!(
                "      <content>{}</content>\n",
                self.escape_xml(&message.content.to_string())
            ));
            if self.config.include_timestamps {
                xml.push_str(&format!(
                    "      <timestamp>{}</timestamp>\n",
                    message.timestamp.to_rfc3339()
                ));
            }
            xml.push_str("    </message>\n");
        }
        xml.push_str("  </messages>\n");

        xml.push_str("</session>\n");

        Ok(xml)
    }

    /// Export to Markdown format
    fn export_markdown(&self, session: &SessionData) -> Result<String> {
        let mut md = String::new();

        md.push_str(&format!("# Chat Session: {}\n\n", session.id));

        if self.config.include_timestamps {
            md.push_str(&format!("Created: {}\n\n", session.created_at));
        }

        md.push_str("## Conversation\n\n");

        for message in &session.messages {
            let role = format!("{:?}", message.role);
            md.push_str(&format!("### {}\n\n", role));

            if self.config.include_timestamps {
                md.push_str(&format!("*{}*\n\n", message.timestamp.to_rfc3339()));
            }

            md.push_str(&message.content.to_string());
            md.push_str("\n\n---\n\n");
        }

        if self.config.include_statistics {
            md.push_str("## Statistics\n\n");
            let metrics = &session.performance_metrics;
            md.push_str(&format!(
                "- **Total Messages**: {}\n",
                metrics.total_messages
            ));
            md.push_str(&format!("- **User Messages**: {}\n", metrics.user_messages));
            md.push_str(&format!(
                "- **Assistant Messages**: {}\n",
                metrics.assistant_messages
            ));
            md.push_str(&format!(
                "- **Average Response Time**: {:.2}s\n",
                metrics.average_response_time
            ));
            md.push_str(&format!(
                "- **Successful Queries**: {}\n",
                metrics.successful_queries
            ));
            md.push_str(&format!(
                "- **Failed Queries**: {}\n",
                metrics.failed_queries
            ));
        }

        Ok(md)
    }

    /// Export to HTML format
    fn export_html(&self, session: &SessionData) -> Result<String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str(&format!("  <title>Chat Session: {}</title>\n", session.id));
        html.push_str("  <style>\n");
        html.push_str("    body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str("    .message { margin: 10px 0; padding: 10px; border-radius: 5px; }\n");
        html.push_str("    .user { background-color: #e3f2fd; }\n");
        html.push_str("    .assistant { background-color: #f3e5f5; }\n");
        html.push_str("    .timestamp { color: #666; font-size: 0.9em; }\n");
        html.push_str("  </style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str(&format!("  <h1>Chat Session: {}</h1>\n", session.id));

        for message in &session.messages {
            let role = format!("{:?}", message.role).to_lowercase();
            html.push_str(&format!("  <div class=\"message {}\">\n", role));
            html.push_str(&format!("    <strong>{:?}</strong>\n", message.role));

            if self.config.include_timestamps {
                html.push_str(&format!(
                    "    <div class=\"timestamp\">{}</div>\n",
                    message.timestamp.to_rfc3339()
                ));
            }

            html.push_str(&format!(
                "    <p>{}</p>\n",
                self.escape_html(&message.content.to_string())
            ));
            html.push_str("  </div>\n");
        }

        html.push_str("</body>\n</html>\n");

        Ok(html)
    }

    /// Export to plain text format
    fn export_text(&self, session: &SessionData) -> Result<String> {
        let mut text = String::new();

        text.push_str(&format!("Chat Session: {}\n", session.id));
        text.push_str(&format!("Created: {}\n", session.created_at));
        text.push('\n');
        text.push_str(&"=".repeat(80));
        text.push('\n');
        text.push('\n');

        for message in &session.messages {
            text.push_str(&format!("[{:?}]\n", message.role));

            if self.config.include_timestamps {
                text.push_str(&format!("Time: {}\n", message.timestamp.to_rfc3339()));
            }

            text.push_str(&message.content.to_string());
            text.push_str("\n\n");
            text.push_str(&"-".repeat(80));
            text.push_str("\n\n");
        }

        Ok(text)
    }

    // Helper methods for message arrays

    fn messages_to_json(&self, messages: &[Message]) -> Result<String> {
        if self.config.pretty_print {
            serde_json::to_string_pretty(messages).context("Failed to serialize messages to JSON")
        } else {
            serde_json::to_string(messages).context("Failed to serialize messages to JSON")
        }
    }

    fn messages_to_csv(&self, messages: &[Message]) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("Role,Content,Timestamp,Message ID\n");

        for message in messages {
            csv.push_str(&format!(
                "{:?},{},{},{}\n",
                message.role,
                self.escape_csv(&message.content.to_string()),
                message.timestamp.to_rfc3339(),
                message.id
            ));
        }

        Ok(csv)
    }

    fn messages_to_xml(&self, messages: &[Message]) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<messages>\n");

        for message in messages {
            xml.push_str("  <message>\n");
            xml.push_str(&format!("    <id>{}</id>\n", message.id));
            xml.push_str(&format!("    <role>{:?}</role>\n", message.role));
            xml.push_str(&format!(
                "    <content>{}</content>\n",
                self.escape_xml(&message.content.to_string())
            ));
            xml.push_str(&format!(
                "    <timestamp>{}</timestamp>\n",
                message.timestamp.to_rfc3339()
            ));
            xml.push_str("  </message>\n");
        }

        xml.push_str("</messages>\n");
        Ok(xml)
    }

    fn messages_to_markdown(&self, messages: &[Message]) -> Result<String> {
        let mut md = String::new();
        md.push_str("# Chat Messages\n\n");

        for message in messages {
            md.push_str(&format!("### {:?}\n\n", message.role));
            md.push_str(&format!("*{}*\n\n", message.timestamp.to_rfc3339()));
            md.push_str(&message.content.to_string());
            md.push_str("\n\n---\n\n");
        }

        Ok(md)
    }

    fn messages_to_html(&self, messages: &[Message]) -> Result<String> {
        let mut html = String::from("<html><body>\n");

        for message in messages {
            html.push_str(&format!("<div class=\"message {:?}\">\n", message.role));
            html.push_str(&format!("  <strong>{:?}</strong>\n", message.role));
            html.push_str(&format!(
                "  <p>{}</p>\n",
                self.escape_html(&message.content.to_string())
            ));
            html.push_str("</div>\n");
        }

        html.push_str("</body></html>\n");
        Ok(html)
    }

    fn messages_to_text(&self, messages: &[Message]) -> Result<String> {
        let mut text = String::new();

        for message in messages {
            text.push_str(&format!("[{:?}] {}\n\n", message.role, message.content));
        }

        Ok(text)
    }

    // Escaping helpers

    fn escape_csv(&self, s: &str) -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    }

    fn escape_xml(&self, s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    fn escape_html(&self, s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }

    /// Export to file
    pub fn export_to_file<P: AsRef<Path>>(
        &self,
        session: &SessionData,
        format: ExportFormat,
        path: P,
    ) -> Result<()> {
        let content = self.export_session(session, format)?;
        std::fs::write(path, content).context("Failed to write export file")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{Message, MessageContent, MessageRole};
    use chrono::Utc;

    fn create_test_message(role: MessageRole, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content: MessageContent::from_text(content.to_string()),
            timestamp: Utc::now(),
            metadata: None,
            thread_id: None,
            parent_message_id: None,
            token_count: None,
            reactions: Vec::new(),
            attachments: Vec::new(),
            rich_elements: Vec::new(),
        }
    }

    #[test]
    fn test_json_export() {
        let exporter = ChatExporter::new(ExportConfig::default());
        let messages = vec![
            create_test_message(MessageRole::User, "Hello"),
            create_test_message(MessageRole::Assistant, "Hi there!"),
        ];

        let json = exporter.messages_to_json(&messages).unwrap();
        assert!(json.contains("Hello"));
        assert!(json.contains("Hi there"));
    }

    #[test]
    fn test_csv_export() {
        let exporter = ChatExporter::new(ExportConfig::default());
        let messages = vec![create_test_message(MessageRole::User, "Hello")];

        let csv = exporter.messages_to_csv(&messages).unwrap();
        assert!(csv.contains("Role,Content"));
        assert!(csv.contains("Hello"));
    }

    #[test]
    fn test_markdown_export() {
        let exporter = ChatExporter::new(ExportConfig::default());
        let messages = vec![create_test_message(MessageRole::User, "Hello")];

        let md = exporter.messages_to_markdown(&messages).unwrap();
        assert!(md.contains("# Chat Messages"));
        assert!(md.contains("Hello"));
    }
}
