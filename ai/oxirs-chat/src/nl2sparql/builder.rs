//! Extended `NL2SPARQLSystem` impl: template management, generation strategies,
//! confidence scoring, and explanation generation.

use anyhow::{anyhow, Result};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, error, info, warn};

use super::{
    types::{
        GenerationMetadata, GenerationMethod, OptimizationHint, OptimizationHintType,
        ParameterType, QueryComplexity, QueryExplanation, ReasoningStep, ReasoningStepType,
        SPARQLGenerationResult, SPARQLTemplate, SemanticWarning, SemanticWarningType,
        TemplateExample, TemplateParameter, ValidationResult, WarningSeverity,
    },
    NL2SPARQLSystem,
};
use crate::{
    llm::{ChatMessage, ChatRole, LLMRequest, Priority, UseCase},
    QueryContext,
};

impl NL2SPARQLSystem {
    pub(crate) fn initialize_templates(&mut self) -> Result<()> {
        self.add_built_in_templates()?;

        if let Some(template_dir) = self.config.templates.template_dir.clone() {
            self.load_templates_from_directory(&template_dir)?;
        }

        Ok(())
    }

    fn add_built_in_templates(&mut self) -> Result<()> {
        let factual_template = SPARQLTemplate {
            name: "factual_lookup".to_string(),
            description: "Simple factual lookup queries".to_string(),
            intent_patterns: vec![
                "what is".to_string(),
                "who is".to_string(),
                "where is".to_string(),
                "when was".to_string(),
            ],
            template: r#"
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?answer WHERE {
    {{entity}} {{property}} ?answer .
    OPTIONAL { ?answer rdfs:label ?label }
}
"#
            .to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "entity".to_string(),
                    parameter_type: ParameterType::Entity,
                    required: true,
                    default_value: None,
                    extraction_pattern: Some(
                        r"(?:what|who|where|when) (?:is|was) (.+?)(?:\?|$)".to_string(),
                    ),
                },
                TemplateParameter {
                    name: "property".to_string(),
                    parameter_type: ParameterType::Property,
                    required: true,
                    default_value: Some("rdfs:label".to_string()),
                    extraction_pattern: None,
                },
            ],
            examples: vec![TemplateExample {
                natural_language: "What is the capital of France?".to_string(),
                parameters: [
                    (
                        "entity".to_string(),
                        "<http://example.org/France>".to_string(),
                    ),
                    (
                        "property".to_string(),
                        "<http://example.org/capital>".to_string(),
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
                expected_sparql:
                    "SELECT ?answer WHERE { <http://example.org/France> <http://example.org/capital> ?answer }"
                        .to_string(),
            }],
            complexity: QueryComplexity::Simple,
        };

        self.templates
            .insert("factual_lookup".to_string(), factual_template);

        let relationship_template = SPARQLTemplate {
            name: "relationship_query".to_string(),
            description: "Queries about relationships between entities".to_string(),
            intent_patterns: vec![
                "how is".to_string(),
                "what is the relationship".to_string(),
                "how are.*related".to_string(),
            ],
            template: r#"
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?path ?relation WHERE {
    {{entity1}} ?relation {{entity2}} .
    OPTIONAL { ?relation rdfs:label ?path }
}
UNION {
    {{entity1}} ?p1 ?intermediate .
    ?intermediate ?p2 {{entity2}} .
    BIND(CONCAT(STR(?p1), " -> ", STR(?p2)) AS ?path)
    BIND(?p1 AS ?relation)
}
"#
            .to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "entity1".to_string(),
                    parameter_type: ParameterType::Entity,
                    required: true,
                    default_value: None,
                    extraction_pattern: Some(
                        r"(?:how|what) (?:is|are) (.+?) (?:related to|connected to) (.+?)"
                            .to_string(),
                    ),
                },
                TemplateParameter {
                    name: "entity2".to_string(),
                    parameter_type: ParameterType::Entity,
                    required: true,
                    default_value: None,
                    extraction_pattern: None,
                },
            ],
            examples: vec![],
            complexity: QueryComplexity::Medium,
        };

        self.templates
            .insert("relationship_query".to_string(), relationship_template);

        let list_template = SPARQLTemplate {
            name: "list_query".to_string(),
            description: "Queries that return lists of items".to_string(),
            intent_patterns: vec![
                "list all".to_string(),
                "show me".to_string(),
                "what are".to_string(),
                "which".to_string(),
            ],
            template: r#"
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT DISTINCT ?item ?label WHERE {
    ?item rdf:type {{type}} .
    {{#if filter}}
    ?item {{filter_property}} {{filter_value}} .
    {{/if}}
    OPTIONAL { ?item rdfs:label ?label }
}
ORDER BY ?label
LIMIT {{limit}}
"#
            .to_string(),
            parameters: vec![
                TemplateParameter {
                    name: "type".to_string(),
                    parameter_type: ParameterType::Class,
                    required: true,
                    default_value: None,
                    extraction_pattern: Some(
                        r"(?:list all|show me|what are) (.+?)(?:\?|$)".to_string(),
                    ),
                },
                TemplateParameter {
                    name: "limit".to_string(),
                    parameter_type: ParameterType::Literal,
                    required: false,
                    default_value: Some("100".to_string()),
                    extraction_pattern: None,
                },
            ],
            examples: vec![],
            complexity: QueryComplexity::Simple,
        };

        self.templates
            .insert("list_query".to_string(), list_template);

        Ok(())
    }

    fn load_templates_from_directory(&mut self, dir: &str) -> Result<()> {
        let dir_path = Path::new(dir);

        if !dir_path.exists() {
            warn!("Template directory does not exist: {}", dir);
            return Ok(());
        }

        info!("Loading templates from directory: {}", dir);

        let entries = fs::read_dir(dir_path)?;
        let mut loaded_count = 0;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let extension = path.extension().and_then(|s| s.to_str());

                match extension {
                    Some("json") => {
                        if let Err(e) = self.load_json_template(&path) {
                            error!("Failed to load JSON template from {:?}: {}", path, e);
                        } else {
                            loaded_count += 1;
                            debug!("Loaded template from {:?}", path);
                        }
                    }
                    Some("yaml") | Some("yml") => {
                        if let Err(e) = self.load_yaml_template(&path) {
                            error!("Failed to load YAML template from {:?}: {}", path, e);
                        } else {
                            loaded_count += 1;
                            debug!("Loaded template from {:?}", path);
                        }
                    }
                    _ => {
                        debug!("Skipping non-template file: {:?}", path);
                    }
                }
            }
        }

        info!("Loaded {} templates from directory: {}", loaded_count, dir);
        Ok(())
    }

    fn load_json_template(&mut self, path: &Path) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let template: SPARQLTemplate = serde_json::from_str(&content)?;

        if template.name.is_empty() {
            return Err(anyhow!("Template name cannot be empty"));
        }
        if template.template.is_empty() {
            return Err(anyhow!("Template SPARQL cannot be empty"));
        }

        self.template_engine
            .register_template_string(&template.name, &template.template)?;
        self.templates.insert(template.name.clone(), template);

        Ok(())
    }

    fn load_yaml_template(&mut self, path: &Path) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let template: SPARQLTemplate = serde_yaml::from_str(&content)?;

        if template.name.is_empty() {
            return Err(anyhow!("Template name cannot be empty"));
        }
        if template.template.is_empty() {
            return Err(anyhow!("Template SPARQL cannot be empty"));
        }

        self.template_engine
            .register_template_string(&template.name, &template.template)?;
        self.templates.insert(template.name.clone(), template);

        Ok(())
    }

    pub(crate) async fn generate_with_templates(
        &self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        let template = self.select_template(query_context)?;
        let parameters = self.extract_parameters(template, query_context)?;
        let sparql_query = self.fill_template(template, &parameters)?;
        let confidence =
            self.calculate_template_confidence(template, &parameters, query_context)?;

        Ok(SPARQLGenerationResult {
            query: sparql_query,
            confidence,
            generation_method: GenerationMethod::Template(template.name.clone()),
            parameters,
            explanation: None,
            validation_result: ValidationResult {
                is_valid: true,
                syntax_errors: Vec::new(),
                semantic_warnings: Vec::new(),
                schema_issues: Vec::new(),
                suggestions: Vec::new(),
            },
            optimization_hints: Vec::new(),
            metadata: GenerationMetadata {
                generation_time_ms: 0,
                template_used: Some(template.name.clone()),
                llm_model_used: None,
                iterations: 1,
                fallback_used: false,
            },
        })
    }

    pub(crate) async fn generate_with_llm(
        &mut self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        let system_prompt = self.create_sparql_generation_prompt();

        if let Some(llm_manager) = self.llm_manager.clone() {
            let query_text = query_context
                .conversation_history
                .iter()
                .rev()
                .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
                .map(|msg| msg.content.as_str())
                .unwrap_or("Unknown query");
            let user_message =
                format!("Convert this natural language query to SPARQL: {query_text}");

            let llm_request = LLMRequest {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: user_message,
                    metadata: None,
                }],
                system_prompt: Some(system_prompt),
                temperature: 0.3,
                max_tokens: Some(1000),
                use_case: UseCase::SparqlGeneration,
                priority: Priority::Normal,
                timeout: None,
            };

            let mut manager = llm_manager.lock().await;
            match manager.generate_response(llm_request).await {
                Ok(response) => {
                    let sparql_query = self.extract_sparql_from_response(&response.content)?;
                    let confidence =
                        self.calculate_llm_confidence(&response, &sparql_query, query_context)?;

                    Ok(SPARQLGenerationResult {
                        query: sparql_query,
                        confidence,
                        generation_method: GenerationMethod::LLM(response.model_used.clone()),
                        parameters: HashMap::new(),
                        explanation: None,
                        validation_result: ValidationResult {
                            is_valid: true,
                            syntax_errors: Vec::new(),
                            semantic_warnings: Vec::new(),
                            schema_issues: Vec::new(),
                            suggestions: Vec::new(),
                        },
                        optimization_hints: Vec::new(),
                        metadata: GenerationMetadata {
                            generation_time_ms: response.latency.as_millis() as u64,
                            template_used: None,
                            llm_model_used: Some(response.model_used),
                            iterations: 1,
                            fallback_used: false,
                        },
                    })
                }
                Err(e) => {
                    warn!("LLM generation failed: {}", e);
                    let mut result = self.generate_with_templates(query_context).await?;
                    result.metadata.fallback_used = true;
                    Ok(result)
                }
            }
        } else {
            self.generate_with_templates(query_context).await
        }
    }

    pub(crate) async fn generate_hybrid(
        &mut self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        let template_result = self.generate_with_templates(query_context).await?;

        if template_result.confidence < self.config.generation.confidence_threshold {
            if let Ok(llm_result) = self.generate_with_llm(query_context).await {
                if llm_result.confidence > template_result.confidence {
                    return Ok(SPARQLGenerationResult {
                        generation_method: GenerationMethod::Hybrid,
                        ..llm_result
                    });
                }
            }
        }

        Ok(SPARQLGenerationResult {
            generation_method: GenerationMethod::Hybrid,
            ..template_result
        })
    }

    pub(crate) async fn generate_rule_based(
        &self,
        query_context: &QueryContext,
    ) -> Result<SPARQLGenerationResult> {
        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("")
            .to_lowercase();

        if query_text.is_empty() {
            return Err(anyhow!("No user query found in context"));
        }

        let mut parameters = HashMap::new();
        let mut confidence: f32 = 0.0;
        let mut hints = Vec::new();

        let select_query = if query_text.contains("find")
            || query_text.contains("show")
            || query_text.contains("list")
            || query_text.contains("get")
            || query_text.contains("what")
            || query_text.contains("which")
        {
            confidence += 0.3;
            true
        } else {
            false
        };

        let count_query = if query_text.contains("how many")
            || query_text.contains("count")
            || query_text.contains("number of")
        {
            confidence += 0.3;
            true
        } else {
            false
        };

        let ask_query = if query_text.starts_with("is")
            || query_text.starts_with("does")
            || query_text.starts_with("has")
            || query_text.contains("whether")
        {
            confidence += 0.3;
            true
        } else {
            false
        };

        let mut subjects = Vec::new();
        let mut predicates = Vec::new();
        let mut objects = Vec::new();

        if let Some(entities) = &query_context.entities {
            for entity in entities {
                if entity.starts_with("http") || entity.starts_with("urn:") {
                    subjects.push(format!("<{entity}>"));
                } else {
                    objects.push(format!("\"{entity}\""));
                }
                parameters.insert(format!("entity_{entity}"), entity.clone());
            }
        }

        if query_text.contains("name") || query_text.contains("label") {
            predicates.push("rdfs:label".to_string());
            confidence += 0.1;
        }
        if query_text.contains("type") || query_text.contains("kind") {
            predicates.push("rdf:type".to_string());
            confidence += 0.1;
        }
        if query_text.contains("born") || query_text.contains("birth") {
            predicates.push("dbo:birthDate".to_string());
            confidence += 0.1;
        }
        if query_text.contains("location") || query_text.contains("place") {
            predicates.push("dbo:location".to_string());
            confidence += 0.1;
        }

        let sparql_query = if count_query {
            let subject_var = if subjects.is_empty() { "?entity" } else { "?s" };
            let predicate = if predicates.is_empty() {
                "?p"
            } else {
                &predicates[0]
            };
            let object_var = if objects.is_empty() {
                "?o"
            } else {
                &objects[0]
            };

            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::SimplifyExpression,
                description: "COUNT queries can be optimized with LIMIT".to_string(),
                estimated_improvement: Some(0.5),
            });

            format!(
                "SELECT (COUNT(*) AS ?count) WHERE {{\n  {subject_var} {predicate} {object_var} .\n}}"
            )
        } else if ask_query {
            let subject = if subjects.is_empty() {
                "?s"
            } else {
                &subjects[0]
            };
            let predicate = if predicates.is_empty() {
                "?p"
            } else {
                &predicates[0]
            };
            let object = if objects.is_empty() {
                "?o"
            } else {
                &objects[0]
            };

            format!("ASK {{\n  {subject} {predicate} {object} .\n}}")
        } else {
            let mut select_vars = Vec::new();
            let mut where_patterns = Vec::new();

            if subjects.is_empty() {
                select_vars.push("?subject");
                where_patterns.push(format!(
                    "?subject {} ?object",
                    if predicates.is_empty() {
                        "?predicate"
                    } else {
                        &predicates[0]
                    }
                ));
            } else {
                select_vars.push("?object");
                where_patterns.push(format!(
                    "{} {} ?object",
                    &subjects[0],
                    if predicates.is_empty() {
                        "?predicate"
                    } else {
                        &predicates[0]
                    }
                ));
            }

            if predicates.is_empty() {
                select_vars.push("?predicate");
            }

            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::UseFilter,
                description: "Consider adding LIMIT clause for large result sets".to_string(),
                estimated_improvement: Some(0.8),
            });

            format!(
                "SELECT {} WHERE {{\n  {}\n}} LIMIT 100",
                select_vars.join(" "),
                where_patterns.join(" .\n  ")
            )
        };

        if !subjects.is_empty() {
            confidence += 0.2;
        }
        if !predicates.is_empty() {
            confidence += 0.2;
        }
        if !objects.is_empty() {
            confidence += 0.1;
        }

        confidence = confidence.clamp(0.0, 1.0);

        let final_query = if confidence < 0.3 {
            confidence = 0.3;
            hints.push(OptimizationHint {
                hint_type: OptimizationHintType::SimplifyExpression,
                description: "Low confidence - using generic query pattern".to_string(),
                estimated_improvement: Some(0.2),
            });
            "SELECT ?subject ?predicate ?object WHERE {\n  ?subject ?predicate ?object .\n} LIMIT 10"
                .to_string()
        } else {
            sparql_query
        };

        let metadata = GenerationMetadata {
            generation_time_ms: 50,
            template_used: None,
            llm_model_used: None,
            iterations: 1,
            fallback_used: confidence < 0.3,
        };

        let explanation = QueryExplanation {
            natural_language: format!(
                "This query was generated using rule-based analysis. Detected patterns: {}, confidence: {:.1}%",
                if select_query { "selection" }
                else if count_query { "counting" }
                else if ask_query { "yes/no question" }
                else { "general query" },
                confidence * 100.0
            ),
            reasoning_steps: vec![
                ReasoningStep {
                    step_type: ReasoningStepType::EntityExtraction,
                    description: "Analyzed natural language for query patterns".to_string(),
                    input: query_text.clone(),
                    output: format!("Detected: select={select_query}, count={count_query}, ask={ask_query}"),
                    confidence: 0.8,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::EntityExtraction,
                    description: "Extracted entities from conversation context".to_string(),
                    input: "Entity groups from context".to_string(),
                    output: format!("Found {} subjects, {} objects", subjects.len(), objects.len()),
                    confidence: 0.7,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::PropertyMapping,
                    description: "Inferred predicates from common vocabulary patterns".to_string(),
                    input: "Common property keywords".to_string(),
                    output: format!("Detected {} predicates", predicates.len()),
                    confidence: 0.6,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::QueryConstruction,
                    description: "Generated SPARQL based on linguistic rules".to_string(),
                    input: "Combined patterns and entities".to_string(),
                    output: "SPARQL query structure".to_string(),
                    confidence,
                },
            ],
            parameter_mapping: parameters.clone(),
            alternatives: vec![
                "Template-based generation for better structure".to_string(),
                "LLM-based generation for complex queries".to_string(),
            ],
        };

        Ok(SPARQLGenerationResult {
            query: final_query,
            confidence,
            generation_method: GenerationMethod::RuleBased,
            parameters,
            explanation: Some(explanation),
            validation_result: ValidationResult {
                is_valid: true,
                syntax_errors: Vec::new(),
                semantic_warnings: if confidence < 0.5 {
                    vec![SemanticWarning {
                        message: "Low confidence rule-based generation".to_string(),
                        warning_type: SemanticWarningType::PerformanceIssue,
                        severity: WarningSeverity::Medium,
                    }]
                } else {
                    Vec::new()
                },
                schema_issues: Vec::new(),
                suggestions: vec![
                    "Consider using template-based or LLM-based generation for better accuracy"
                        .to_string(),
                ],
            },
            optimization_hints: hints,
            metadata,
        })
    }

    fn select_template(&self, query_context: &QueryContext) -> Result<&SPARQLTemplate> {
        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("Unknown query");
        let query_lower = query_text.to_lowercase();

        for template in self.templates.values() {
            for pattern in &template.intent_patterns {
                if query_lower.contains(pattern) {
                    return Ok(template);
                }
            }
        }

        self.templates
            .get("factual_lookup")
            .ok_or_else(|| anyhow!("No suitable template found"))
    }

    fn extract_parameters(
        &self,
        template: &SPARQLTemplate,
        query_context: &QueryContext,
    ) -> Result<HashMap<String, String>> {
        let mut parameters = HashMap::new();
        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("Unknown query");

        for param in &template.parameters {
            if let Some(ref pattern) = param.extraction_pattern {
                if let Ok(regex) = Regex::new(pattern) {
                    if let Some(captures) = regex.captures(query_text) {
                        if let Some(captured) = captures.get(1) {
                            parameters.insert(param.name.clone(), captured.as_str().to_string());
                        }
                    }
                }
            }

            if !parameters.contains_key(&param.name) {
                if let Some(ref default) = param.default_value {
                    parameters.insert(param.name.clone(), default.clone());
                } else if param.required {
                    return Err(anyhow!("Required parameter '{}' not found", param.name));
                }
            }
        }

        Ok(parameters)
    }

    fn fill_template(
        &self,
        template: &SPARQLTemplate,
        parameters: &HashMap<String, String>,
    ) -> Result<String> {
        let template_obj = self
            .template_engine
            .render_template(&template.template, &parameters)?;
        Ok(template_obj)
    }

    pub(crate) fn create_sparql_generation_prompt(&self) -> String {
        let mut prompt =
            r#"You are an expert at converting natural language queries to SPARQL queries.

Guidelines:
1. Generate valid SPARQL 1.1 syntax
2. Use appropriate prefixes (rdf, rdfs, owl, etc.)
3. Include OPTIONAL clauses for optional data
4. Use FILTER clauses for constraints
5. Add LIMIT clauses for list queries
6. Use proper variable naming
7. Include comments explaining complex parts

"#
            .to_string();

        if let Some(schema) = &self.schema {
            prompt.push_str("\n**Available Schema Information:**\n\n");

            if !schema.prefixes.is_empty() {
                prompt.push_str("**Common Prefixes:**\n");
                for (prefix, uri) in schema.prefixes.iter().take(10) {
                    prompt.push_str(&format!("PREFIX {}: <{}>\n", prefix, uri));
                }
                prompt.push('\n');
            }

            if !schema.classes.is_empty() {
                prompt.push_str("**Available Classes:**\n");
                for class in schema.classes.iter().take(15) {
                    if let Some(label) = &class.label {
                        prompt.push_str(&format!(
                            "- {} ({}): {} instances\n",
                            label, class.uri, class.instance_count
                        ));
                    } else {
                        prompt.push_str(&format!(
                            "- {}: {} instances\n",
                            class.uri, class.instance_count
                        ));
                    }
                    if !class.properties.is_empty() {
                        let key_props: Vec<String> = class
                            .properties
                            .iter()
                            .take(5)
                            .map(|p| {
                                p.label.clone().unwrap_or_else(|| {
                                    p.uri
                                        .split(&['#', '/'][..])
                                        .next_back()
                                        .unwrap_or(&p.uri)
                                        .to_string()
                                })
                            })
                            .collect();
                        prompt.push_str(&format!("  Properties: {}\n", key_props.join(", ")));
                    }
                }
                prompt.push('\n');
            }

            if !schema.properties.is_empty() {
                prompt.push_str("**Frequently Used Properties:**\n");
                for property in schema.properties.iter().take(20) {
                    if let Some(label) = &property.label {
                        prompt.push_str(&format!(
                            "- {} ({}): {} usages\n",
                            label, property.uri, property.usage_count
                        ));
                    } else {
                        prompt.push_str(&format!(
                            "- {}: {} usages\n",
                            property.uri, property.usage_count
                        ));
                    }
                }
                prompt.push('\n');
            }

            prompt.push_str("Use the schema information above to generate accurate SPARQL queries with correct class and property URIs.\n\n");
        }

        prompt.push_str("Always respond with just the SPARQL query, no additional explanation unless requested.");
        prompt
    }

    pub(crate) fn extract_sparql_from_response(&self, response: &str) -> Result<String> {
        if let Some(start) = response.find("```sparql") {
            if let Some(end) = response[start..].find("```") {
                let query = &response[start + 9..start + end];
                return Ok(query.trim().to_string());
            }
        }

        if let Some(start) = response.find("```") {
            if let Some(end) = response[start + 3..].find("```") {
                let query = &response[start + 3..start + 3 + end];
                return Ok(query.trim().to_string());
            }
        }

        let trimmed = response.trim();
        if trimmed.to_uppercase().contains("SELECT")
            || trimmed.to_uppercase().contains("CONSTRUCT")
            || trimmed.to_uppercase().contains("ASK")
            || trimmed.to_uppercase().contains("DESCRIBE")
        {
            return Ok(trimmed.to_string());
        }

        Err(anyhow!("Could not extract SPARQL query from response"))
    }

    pub(crate) async fn generate_explanation(
        &self,
        result: &SPARQLGenerationResult,
        query_context: &QueryContext,
    ) -> Result<QueryExplanation> {
        let query_text = query_context
            .conversation_history
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
            .map(|msg| msg.content.as_str())
            .unwrap_or("Unknown query");

        let mut reasoning_steps = Vec::new();
        let mut alternatives = Vec::new();

        reasoning_steps.push(ReasoningStep {
            step_type: ReasoningStepType::EntityExtraction,
            description: "Analyzed natural language query".to_string(),
            input: query_text.to_string(),
            output: format!("Identified intent: {:?}", query_context.query_intent),
            confidence: 0.9,
        });

        let method_description = match result.generation_method {
            GenerationMethod::Template(ref template_name) => {
                format!("Selected template-based generation using template: {template_name}")
            }
            GenerationMethod::LLM(ref model_name) => {
                format!("Selected LLM-based generation using model: {model_name}")
            }
            GenerationMethod::Hybrid => {
                "Selected hybrid approach combining template and LLM generation".to_string()
            }
            GenerationMethod::RuleBased => "Selected rule-based generation approach".to_string(),
        };

        reasoning_steps.push(ReasoningStep {
            step_type: ReasoningStepType::TemplateSelection,
            description: method_description,
            input: "Query analysis results".to_string(),
            output: format!("Generation method: {:?}", result.generation_method),
            confidence: result.confidence,
        });

        if !result.parameters.is_empty() {
            let parameters_description = result
                .parameters
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(", ");

            reasoning_steps.push(ReasoningStep {
                step_type: ReasoningStepType::ParameterFilling,
                description: "Extracted parameters from natural language".to_string(),
                input: query_text.to_string(),
                output: parameters_description,
                confidence: 0.8,
            });
        }

        reasoning_steps.push(ReasoningStep {
            step_type: ReasoningStepType::QueryConstruction,
            description: "Constructed SPARQL query from parameters".to_string(),
            input: "Template and extracted parameters".to_string(),
            output: result.query.clone(),
            confidence: result.confidence,
        });

        if result.validation_result.is_valid {
            reasoning_steps.push(ReasoningStep {
                step_type: ReasoningStepType::Validation,
                description: "Query validated successfully".to_string(),
                input: result.query.clone(),
                output: "Query is syntactically and semantically valid".to_string(),
                confidence: 1.0,
            });
        } else {
            reasoning_steps.push(ReasoningStep {
                step_type: ReasoningStepType::Validation,
                description: "Query validation found issues".to_string(),
                input: result.query.clone(),
                output: format!(
                    "Found {} errors and {} warnings",
                    result.validation_result.syntax_errors.len(),
                    result.validation_result.semantic_warnings.len()
                ),
                confidence: 0.5,
            });
        }

        if !result.optimization_hints.is_empty() {
            reasoning_steps.push(ReasoningStep {
                step_type: ReasoningStepType::Optimization,
                description: "Applied query optimizations".to_string(),
                input: "Original query".to_string(),
                output: format!(
                    "Applied {} optimization hints",
                    result.optimization_hints.len()
                ),
                confidence: 0.9,
            });
        }

        alternatives.push("Could have used different parameter extraction patterns".to_string());
        alternatives
            .push("Could have selected a different template or generation method".to_string());

        if result.confidence < 0.8 {
            alternatives.push("Consider rephrasing the query for better accuracy".to_string());
        }

        let natural_language = self
            .generate_natural_language_explanation(query_text, result, &reasoning_steps)
            .await?;

        Ok(QueryExplanation {
            natural_language,
            reasoning_steps,
            parameter_mapping: result.parameters.clone(),
            alternatives,
        })
    }

    async fn generate_natural_language_explanation(
        &self,
        query_text: &str,
        result: &SPARQLGenerationResult,
        reasoning_steps: &[ReasoningStep],
    ) -> Result<String> {
        let mut explanation = String::new();

        explanation.push_str(&format!("For your query '{query_text}', I:\n\n"));

        for (i, step) in reasoning_steps.iter().enumerate() {
            explanation.push_str(&format!("{}. {}\n", i + 1, step.description));

            if step.confidence < 0.7 {
                explanation.push_str(&format!(
                    "   (Note: This step has lower confidence: {:.1}%)\n",
                    step.confidence * 100.0
                ));
            }
        }

        explanation.push_str(&format!(
            "\nThe resulting SPARQL query has a confidence score of {:.1}%.\n",
            result.confidence * 100.0
        ));

        if result.confidence < 0.7 {
            explanation.push_str("You may want to rephrase your query for better results.\n");
        }

        if !result.validation_result.is_valid {
            explanation.push_str(
                "Note: The generated query has some validation issues that may affect execution.\n",
            );
        }

        Ok(explanation)
    }

    fn calculate_template_confidence(
        &self,
        template: &SPARQLTemplate,
        parameters: &HashMap<String, String>,
        query_context: &QueryContext,
    ) -> Result<f32> {
        let mut confidence_factors = Vec::new();

        let specificity_score = if !template.parameters.is_empty() {
            0.6 + (template.parameters.len() as f32 * 0.1).min(0.4)
        } else {
            0.6
        };
        confidence_factors.push(specificity_score);

        let mut param_quality: f32 = 1.0;
        for param in &template.parameters {
            if let Some(value) = parameters.get(&param.name) {
                if value.is_empty() {
                    param_quality *= 0.7;
                } else if value.len() < 2 {
                    param_quality *= 0.8;
                } else if value.contains("unknown") || value.contains("undefined") {
                    param_quality *= 0.6;
                }
            } else {
                param_quality *= 0.5;
            }
        }
        confidence_factors.push(param_quality.max(0.5));

        let intent_match_score = if template.intent_patterns.is_empty() {
            0.7
        } else {
            let query_text = query_context
                .conversation_history
                .iter()
                .rev()
                .find(|msg| matches!(msg.role, crate::rag::types::MessageRole::User))
                .map(|msg| msg.content.as_str())
                .unwrap_or("");
            let mut best_match: f32 = 0.0;

            for pattern in &template.intent_patterns {
                let pattern_lower = pattern.to_lowercase();
                let pattern_words: std::collections::HashSet<_> =
                    pattern_lower.split_whitespace().collect();
                let query_lower = query_text.to_lowercase();
                let query_words: std::collections::HashSet<_> =
                    query_lower.split_whitespace().collect();

                let intersection_size = pattern_words.intersection(&query_words).count();
                let union_size = pattern_words.union(&query_words).count();

                if union_size > 0 {
                    let jaccard_similarity = intersection_size as f32 / union_size as f32;
                    best_match = best_match.max(jaccard_similarity);
                }
            }

            0.4 + best_match * 0.6
        };
        confidence_factors.push(intent_match_score);

        let base_confidence =
            confidence_factors.iter().sum::<f32>() / confidence_factors.len() as f32;
        let template_bonus = if !template.examples.is_empty() {
            0.05
        } else {
            0.0
        };
        let final_confidence = (base_confidence + template_bonus).clamp(0.1, 1.0);

        debug!(
            "Template confidence for '{}': {:.3}",
            template.name, final_confidence
        );

        Ok(final_confidence)
    }

    fn calculate_llm_confidence(
        &mut self,
        _response: &crate::llm::LLMResponse,
        sparql_query: &str,
        _query_context: &QueryContext,
    ) -> Result<f32> {
        let mut confidence_factors = Vec::new();

        let llm_confidence: f32 = 0.7;
        confidence_factors.push(llm_confidence.max(0.3));

        let syntax_score = if self
            .validator
            .validate(sparql_query)
            .map(|r| r.is_valid)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.2
        };
        confidence_factors.push(syntax_score);

        let completeness_score = if sparql_query.trim().is_empty() {
            0.4
        } else if (sparql_query.to_uppercase().contains("SELECT")
            || sparql_query.to_uppercase().contains("CONSTRUCT"))
            && sparql_query.to_uppercase().contains("WHERE")
        {
            0.9
        } else if sparql_query.to_uppercase().contains("ASK") {
            0.8
        } else {
            0.6
        };
        confidence_factors.push(completeness_score);

        let weights = [0.25_f32, 0.35, 0.40];
        let weighted_sum: f32 = confidence_factors
            .iter()
            .zip(weights.iter())
            .map(|(factor, weight)| factor * weight)
            .sum();

        let final_confidence = weighted_sum.clamp(0.1, 1.0);

        debug!("LLM confidence for query: {:.3}", final_confidence);

        Ok(final_confidence)
    }
}
