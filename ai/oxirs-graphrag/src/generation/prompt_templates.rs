//! Prompt templates for GraphRAG

use serde::{Deserialize, Serialize};

/// Prompt template for LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Template name
    pub name: String,
    /// System prompt
    pub system: String,
    /// User prompt template (with placeholders)
    pub user_template: String,
    /// Placeholders in the template
    pub placeholders: Vec<String>,
}

impl PromptTemplate {
    /// Create default GraphRAG prompt
    pub fn graphrag_default() -> Self {
        Self {
            name: "graphrag_default".to_string(),
            system: r#"You are a knowledge graph assistant. Your task is to answer questions based on the provided knowledge graph context.

Guidelines:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Cite specific entities and relationships when possible
4. Be concise but comprehensive
5. Use natural language, not technical graph notation"#.to_string(),
            user_template: r#"Based on the following knowledge graph context, please answer the question.

{context}

Question: {query}

Please provide a clear and accurate answer based on the knowledge graph facts above."#.to_string(),
            placeholders: vec!["context".to_string(), "query".to_string()],
        }
    }

    /// Create analytical prompt for complex questions
    pub fn graphrag_analytical() -> Self {
        Self {
            name: "graphrag_analytical".to_string(),
            system: r#"You are an expert knowledge graph analyst. Your task is to perform deep analysis of knowledge graph data to answer complex questions.

Guidelines:
1. Analyze relationships and patterns in the data
2. Consider indirect connections and implications
3. Provide reasoning chains when appropriate
4. Highlight any uncertainties or gaps in the data
5. Structure your answer with clear sections if needed"#.to_string(),
            user_template: r#"Perform a detailed analysis of the following knowledge graph data to answer the question.

## Context
{context}

## Community Insights
{communities}

## Question
{query}

Please provide a comprehensive analysis including:
1. Direct answer to the question
2. Supporting evidence from the knowledge graph
3. Any relevant patterns or relationships discovered
4. Confidence level in your answer"#.to_string(),
            placeholders: vec!["context".to_string(), "communities".to_string(), "query".to_string()],
        }
    }

    /// Create summarization prompt
    pub fn graphrag_summarize() -> Self {
        Self {
            name: "graphrag_summarize".to_string(),
            system: r#"You are a knowledge graph summarizer. Your task is to create concise summaries of knowledge graph subgraphs.

Guidelines:
1. Identify the main entities and their relationships
2. Group related facts together
3. Use clear, natural language
4. Preserve important details while being concise
5. Highlight any notable patterns"#.to_string(),
            user_template: r#"Please summarize the following knowledge graph data:

{context}

Create a clear summary that captures the main entities, relationships, and key insights."#.to_string(),
            placeholders: vec!["context".to_string()],
        }
    }

    /// Render template with values
    pub fn render(&self, values: &std::collections::HashMap<String, String>) -> String {
        let mut result = self.user_template.clone();

        for placeholder in &self.placeholders {
            if let Some(value) = values.get(placeholder) {
                result = result.replace(&format!("{{{}}}", placeholder), value);
            }
        }

        result
    }

    /// Get full prompt including system message
    pub fn get_full_prompt(
        &self,
        values: &std::collections::HashMap<String, String>,
    ) -> (String, String) {
        (self.system.clone(), self.render(values))
    }
}

/// Prompt builder for custom templates
pub struct PromptBuilder {
    template: PromptTemplate,
}

impl PromptBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            template: PromptTemplate {
                name: name.to_string(),
                system: String::new(),
                user_template: String::new(),
                placeholders: Vec::new(),
            },
        }
    }

    pub fn system(mut self, system: &str) -> Self {
        self.template.system = system.to_string();
        self
    }

    pub fn user_template(mut self, template: &str) -> Self {
        self.template.user_template = template.to_string();
        // Extract placeholders
        let re = regex::Regex::new(r"\{(\w+)\}").unwrap();
        self.template.placeholders = re
            .captures_iter(template)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();
        self
    }

    pub fn build(self) -> PromptTemplate {
        self.template
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_template_rendering() {
        let template = PromptTemplate::graphrag_default();

        let mut values = HashMap::new();
        values.insert(
            "context".to_string(),
            "Entity A is related to Entity B".to_string(),
        );
        values.insert("query".to_string(), "What is A?".to_string());

        let rendered = template.render(&values);

        assert!(rendered.contains("Entity A is related to Entity B"));
        assert!(rendered.contains("What is A?"));
    }

    #[test]
    fn test_prompt_builder() {
        let template = PromptBuilder::new("custom")
            .system("You are a helpful assistant.")
            .user_template("Context: {context}\nQuestion: {query}")
            .build();

        assert_eq!(template.name, "custom");
        assert!(template.placeholders.contains(&"context".to_string()));
        assert!(template.placeholders.contains(&"query".to_string()));
    }
}
