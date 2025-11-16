//! Turtle Parser for SAMM Models

use crate::error::{Result, SammError, SourceLocation};
use crate::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Entity, Event, ModelElement,
    Operation, Property,
};
use oxrdf::{Graph, NamedNode, NamedNodeRef, NamedOrBlankNode, NamedOrBlankNodeRef, Term, TermRef};
use oxttl::TurtleParser;
use std::collections::HashMap;
use std::path::Path;

/// SAMM namespace URI base (version-agnostic)
const SAMM_NS_BASE: &str = "urn:samm:org.eclipse.esmf.samm:meta-model:";
const SAMM_C_NS_BASE: &str = "urn:samm:org.eclipse.esmf.samm:characteristic:";
const SAMM_E_NS_BASE: &str = "urn:samm:org.eclipse.esmf.samm:entity:";

/// Supported SAMM versions
const SAMM_VERSIONS: &[&str] = &["2.3.0", "2.2.0", "2.1.0", "2.0.0"];

/// Parser for Turtle/RDF files containing SAMM models
pub struct SammTurtleParser {
    /// RDF graph containing parsed triples
    graph: Graph,

    /// Cache of parsed elements by URN
    element_cache: HashMap<String, CachedElement>,

    /// Detected SAMM namespace (version-specific)
    samm_ns: Option<String>,

    /// Detected SAMM characteristic namespace
    samm_c_ns: Option<String>,

    /// Detected SAMM entity namespace
    samm_e_ns: Option<String>,

    /// Current source being parsed (file path or description)
    current_source: Option<String>,
}

/// Cached parsed element
enum CachedElement {
    Aspect(Aspect),
    Property(Property),
    Characteristic(Characteristic),
    Entity(Entity),
    Operation(Operation),
    Event(Event),
}

impl SammTurtleParser {
    /// Create a new Turtle parser
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            element_cache: HashMap::new(),
            samm_ns: None,
            samm_c_ns: None,
            samm_e_ns: None,
            current_source: None,
        }
    }

    /// Parse a SAMM model from a Turtle file
    pub async fn parse_file(&mut self, path: &Path) -> Result<Aspect> {
        tracing::info!("Parsing SAMM model from file: {:?}", path);

        // Set current source for error reporting
        self.current_source = Some(path.display().to_string());

        // Read file content
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(SammError::Io)?;

        // Parse as Turtle
        self.parse_string(&content, &format!("file://{}", path.display()))
            .await
    }

    /// Parse a SAMM model from a Turtle string
    pub async fn parse_string(&mut self, content: &str, base_uri: &str) -> Result<Aspect> {
        tracing::debug!("Parsing SAMM model from string, base: {}", base_uri);

        // Parse Turtle into RDF graph
        let parser = TurtleParser::new().with_base_iri(base_uri).map_err(|e| {
            self.create_parse_error(
                &format!("Invalid base URI '{}': {}", base_uri, e),
                None,
                None,
            )
        })?;

        for result in parser.for_reader(content.as_bytes()) {
            let triple = result.map_err(|e| {
                // Try to extract line number from error message if available
                let error_msg = e.to_string();
                let (line, col) = Self::extract_line_col_from_error(&error_msg);
                self.create_parse_error(&error_msg, line, col)
            })?;
            self.graph.insert(&triple);
        }

        tracing::debug!("Parsed {} triples", self.graph.len());

        // Detect SAMM namespace versions
        self.detect_and_set_namespaces();

        // Find the Aspect in the graph
        self.find_and_parse_aspect()
    }

    /// Create a parse error with source location
    fn create_parse_error(
        &self,
        message: &str,
        line: Option<usize>,
        column: Option<usize>,
    ) -> SammError {
        SammError::ParseErrorWithLocation {
            message: message.to_string(),
            location: SourceLocation {
                line,
                column,
                source: self.current_source.clone(),
            },
        }
    }

    /// Try to extract line and column numbers from error message
    /// Many parsers include this info in format like "at line 5, column 10"
    fn extract_line_col_from_error(error_msg: &str) -> (Option<usize>, Option<usize>) {
        // Try to find patterns like "line 5" or "line:5" or "@5:10"
        let line = error_msg.split_whitespace().find_map(|word| {
            if word.starts_with("line") || word.starts_with("Line") {
                word.split(':').next_back()?.parse().ok()
            } else {
                None
            }
        });

        let column = error_msg.split_whitespace().find_map(|word| {
            if word.starts_with("column") || word.starts_with("Column") {
                word.split(':').next_back()?.parse().ok()
            } else {
                None
            }
        });

        (line, column)
    }

    /// Detect and set the SAMM namespaces used in the graph
    fn detect_and_set_namespaces(&mut self) {
        // Try to find any SAMM namespace in the graph
        let mut detected_version = None;

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(s) = triple.subject {
                let s_str = s.as_str();
                for version in SAMM_VERSIONS {
                    if s_str.contains(&format!("{}{}#", SAMM_NS_BASE, version)) {
                        detected_version = Some(version.to_string());
                        break;
                    }
                }
            }
            if detected_version.is_none() {
                if let TermRef::NamedNode(o) = triple.object {
                    let o_str = o.as_str();
                    for version in SAMM_VERSIONS {
                        if o_str.contains(&format!("{}{}#", SAMM_NS_BASE, version)) {
                            detected_version = Some(version.to_string());
                            break;
                        }
                    }
                }
            }
            if detected_version.is_some() {
                break;
            }
        }

        // Set namespaces based on detected version (default to 2.3.0)
        let version = detected_version.unwrap_or_else(|| "2.3.0".to_string());
        self.samm_ns = Some(format!("{}{}#", SAMM_NS_BASE, version));
        self.samm_c_ns = Some(format!("{}{}#", SAMM_C_NS_BASE, version));
        self.samm_e_ns = Some(format!("{}{}#", SAMM_E_NS_BASE, version));

        tracing::debug!("Detected SAMM version: {}", version);
    }

    /// Get the SAMM namespace, ensuring it has been detected
    fn samm_ns(&mut self) -> &str {
        if self.samm_ns.is_none() {
            self.detect_and_set_namespaces();
        }
        self.samm_ns.as_ref().unwrap()
    }

    /// Find and parse the Aspect from the graph
    fn find_and_parse_aspect(&mut self) -> Result<Aspect> {
        // Detect SAMM version from the graph if not already done
        if self.samm_ns.is_none() {
            self.detect_and_set_namespaces();
        }

        let samm_ns = self
            .samm_ns
            .as_ref()
            .ok_or_else(|| SammError::ParseError("SAMM namespace not set".to_string()))?;

        // Find subjects that are of type samm:Aspect
        let aspect_type = NamedNode::new(format!("{}Aspect", samm_ns))
            .map_err(|e| SammError::ParseError(e.to_string()))?;
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        let aspects: Vec<_> = self
            .graph
            .triples_for_predicate(&rdf_type)
            .filter(|triple| {
                if let TermRef::NamedNode(obj) = triple.object {
                    obj == aspect_type.as_ref()
                } else {
                    false
                }
            })
            .collect();

        if aspects.is_empty() {
            return Err(SammError::ParseError(
                "No Aspect found in the model".to_string(),
            ));
        }

        if aspects.len() > 1 {
            tracing::warn!("Multiple Aspects found, using the first one");
        }

        // Get the Aspect subject
        let aspect_subject = aspects[0].subject;
        let aspect_urn = match aspect_subject {
            NamedOrBlankNodeRef::NamedNode(node) => node.as_str().to_string(),
            _ => {
                return Err(SammError::ParseError(
                    "Aspect must be a named node".to_string(),
                ))
            }
        };

        tracing::debug!("Found Aspect: {}", aspect_urn);

        // Parse the Aspect
        self.parse_aspect(&aspect_urn)
    }

    /// Parse an Aspect from the graph
    fn parse_aspect(&mut self, urn: &str) -> Result<Aspect> {
        let subject = NamedNode::new(urn)
            .map_err(|e| SammError::ParseError(format!("Invalid URN '{}': {}", urn, e)))?;

        let mut aspect = Aspect::new(urn.to_string());

        // Parse metadata (preferredName, description, see)
        self.parse_element_metadata(&subject, &mut aspect.metadata)?;

        // Parse properties
        let properties_pred =
            NamedNode::new(format!("{}properties", self.samm_ns.as_ref().unwrap()))
                .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(props_list) = self.get_object(&subject, &properties_pred) {
            let property_urns = self.parse_rdf_list(&props_list)?;
            for prop_urn in property_urns {
                let property = self.parse_property(&prop_urn)?;
                aspect.add_property(property);
            }
        }

        // Parse operations
        let operations_pred =
            NamedNode::new(format!("{}operations", self.samm_ns.as_ref().unwrap()))
                .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(ops_list) = self.get_object(&subject, &operations_pred) {
            let operation_urns = self.parse_rdf_list(&ops_list)?;
            for op_urn in operation_urns {
                let operation = self.parse_operation(&op_urn)?;
                aspect.add_operation(operation);
            }
        }

        // Parse events
        let events_pred = NamedNode::new(format!("{}events", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(events_list) = self.get_object(&subject, &events_pred) {
            let event_urns = self.parse_rdf_list(&events_list)?;
            for event_urn in event_urns {
                let event = self.parse_event(&event_urn)?;
                aspect.add_event(event);
            }
        }

        Ok(aspect)
    }

    /// Parse a Property from the graph
    fn parse_property(&mut self, urn: &str) -> Result<Property> {
        // Check cache
        if let Some(CachedElement::Property(prop)) = self.element_cache.get(urn) {
            return Ok(prop.clone());
        }

        let subject = NamedNode::new(urn)
            .map_err(|e| SammError::ParseError(format!("Invalid URN '{}': {}", urn, e)))?;

        let mut property = Property::new(urn.to_string());

        // Parse metadata
        self.parse_element_metadata(&subject, &mut property.metadata)?;

        // Parse characteristic
        let char_pred = NamedNode::new(format!("{}characteristic", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(char_term) = self.get_object(&subject, &char_pred) {
            let char_urn = self.term_to_string(&char_term)?;
            let characteristic = self.parse_characteristic(&char_urn)?;
            property.characteristic = Some(characteristic);
        }

        // Parse optional flag
        let optional_pred = NamedNode::new(format!("{}optional", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(Term::Literal(lit)) = self.get_object(&subject, &optional_pred) {
            property.optional = lit.value() == "true";
        }

        // Cache the property
        self.element_cache
            .insert(urn.to_string(), CachedElement::Property(property.clone()));

        Ok(property)
    }

    /// Parse a Characteristic from the graph
    fn parse_characteristic(&mut self, urn: &str) -> Result<Characteristic> {
        // Check cache
        if let Some(CachedElement::Characteristic(char)) = self.element_cache.get(urn) {
            return Ok(char.clone());
        }

        // Check if this is a built-in SAMM characteristic (from samm-c namespace)
        let is_builtin = urn.contains(":characteristic:");

        if is_builtin {
            // Extract the characteristic type from the URN
            let type_name = urn.split('#').next_back().unwrap_or("Trait");
            let kind = match type_name {
                "Text" => CharacteristicKind::Trait,
                "Boolean" => CharacteristicKind::Trait,
                "Timestamp" => CharacteristicKind::Trait,
                "Measurement" => CharacteristicKind::Measurement {
                    unit: "unit:one".to_string(),
                },
                "Enumeration" => CharacteristicKind::Enumeration { values: vec![] },
                "State" => CharacteristicKind::State {
                    values: vec![],
                    default_value: None,
                },
                "List" => CharacteristicKind::List {
                    element_characteristic: None,
                },
                "Set" => CharacteristicKind::Set {
                    element_characteristic: None,
                },
                "Collection" => CharacteristicKind::Collection {
                    element_characteristic: None,
                },
                "TimeSeries" => CharacteristicKind::TimeSeries {
                    element_characteristic: None,
                },
                _ => CharacteristicKind::Trait,
            };

            let characteristic = Characteristic::new(urn.to_string(), kind);

            // Cache the built-in characteristic
            self.element_cache.insert(
                urn.to_string(),
                CachedElement::Characteristic(characteristic.clone()),
            );

            return Ok(characteristic);
        }

        let subject = NamedNode::new(urn)
            .map_err(|e| SammError::ParseError(format!("Invalid URN '{}': {}", urn, e)))?;

        // Determine characteristic type
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        let char_type = self
            .get_object(&subject, &rdf_type)
            .ok_or_else(|| SammError::ParseError(format!("No type for characteristic {}", urn)))?;

        let type_str = self.term_to_string(&char_type)?;
        let kind = self.determine_characteristic_kind(&type_str, &subject)?;

        let mut characteristic = Characteristic::new(urn.to_string(), kind);

        // Parse metadata
        self.parse_element_metadata(&subject, &mut characteristic.metadata)?;

        // Parse dataType
        let datatype_pred = NamedNode::new(format!("{}dataType", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(datatype_term) = self.get_object(&subject, &datatype_pred) {
            characteristic.data_type = Some(self.term_to_string(&datatype_term)?);
        }

        // Cache the characteristic
        self.element_cache.insert(
            urn.to_string(),
            CachedElement::Characteristic(characteristic.clone()),
        );

        Ok(characteristic)
    }

    /// Determine the kind of characteristic from its type
    fn determine_characteristic_kind(
        &mut self,
        type_str: &str,
        subject: &NamedNode,
    ) -> Result<CharacteristicKind> {
        // Extract the local name from the type URI
        let local_name = type_str
            .split('#')
            .next_back()
            .unwrap_or(type_str)
            .to_string();

        match local_name.as_str() {
            "Trait" | "Characteristic" => Ok(CharacteristicKind::Trait),
            "Measurement" => {
                // Parse unit from the graph
                let unit = self.parse_unit_from_characteristic(subject)?;
                Ok(CharacteristicKind::Measurement {
                    unit: unit.unwrap_or_else(|| "unit:one".to_string()),
                })
            }
            "Enumeration" => {
                // Parse enumeration values from the graph
                let values = self.parse_enumeration_values(subject)?;
                Ok(CharacteristicKind::Enumeration { values })
            }
            "State" => {
                // Parse state values and default value from the graph
                let values = self.parse_enumeration_values(subject)?;
                let default_value = self.parse_default_value(subject)?;
                Ok(CharacteristicKind::State {
                    values,
                    default_value,
                })
            }
            "List" => Ok(CharacteristicKind::List {
                element_characteristic: None,
            }),
            "Set" => Ok(CharacteristicKind::Set {
                element_characteristic: None,
            }),
            "Collection" => Ok(CharacteristicKind::Collection {
                element_characteristic: None,
            }),
            "TimeSeries" => Ok(CharacteristicKind::TimeSeries {
                element_characteristic: None,
            }),
            _ => {
                tracing::warn!("Unknown characteristic type '{}', using Trait", local_name);
                Ok(CharacteristicKind::Trait)
            }
        }
    }

    /// Parse unit from a Measurement characteristic
    fn parse_unit_from_characteristic(&self, subject: &NamedNode) -> Result<Option<String>> {
        // Try both samm-c:unit and samm:unit predicates
        let unit_pred_c = NamedNode::new(format!("{}unit", self.samm_c_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;
        let unit_pred = NamedNode::new(format!("{}unit", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        // Try samm-c:unit first
        if let Some(unit_term) = self.get_object(subject, &unit_pred_c) {
            return Ok(Some(self.term_to_string(&unit_term)?));
        }

        // Try samm:unit
        if let Some(unit_term) = self.get_object(subject, &unit_pred) {
            return Ok(Some(self.term_to_string(&unit_term)?));
        }

        Ok(None)
    }

    /// Parse enumeration values from the graph
    fn parse_enumeration_values(&self, subject: &NamedNode) -> Result<Vec<String>> {
        // Parse samm-c:values predicate (RDF list of values)
        let values_pred = NamedNode::new(format!("{}values", self.samm_c_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(values_term) = self.get_object(subject, &values_pred) {
            // Parse the RDF list
            return self.parse_rdf_list(&values_term);
        }

        Ok(vec![])
    }

    /// Parse default value from a State characteristic
    fn parse_default_value(&self, subject: &NamedNode) -> Result<Option<String>> {
        let default_pred =
            NamedNode::new(format!("{}defaultValue", self.samm_c_ns.as_ref().unwrap()))
                .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(default_term) = self.get_object(subject, &default_pred) {
            return Ok(Some(self.term_to_string(&default_term)?));
        }

        Ok(None)
    }

    /// Parse an Operation from the graph
    fn parse_operation(&mut self, urn: &str) -> Result<Operation> {
        let subject = NamedNode::new(urn)
            .map_err(|e| SammError::ParseError(format!("Invalid URN '{}': {}", urn, e)))?;

        let mut operation = Operation::new(urn.to_string());

        // Parse metadata
        self.parse_element_metadata(&subject, &mut operation.metadata)?;

        // Parse input parameters (RDF list)
        let input_pred = NamedNode::new(format!("{}input", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(input_term) = self.get_object(&subject, &input_pred) {
            let input_urns = self.parse_rdf_list(&input_term)?;
            for input_urn in input_urns {
                match self.parse_property(&input_urn) {
                    Ok(property) => operation.add_input(property),
                    Err(e) => {
                        tracing::warn!("Failed to parse input property '{}': {}", input_urn, e);
                    }
                }
            }
        }

        // Parse output (single property)
        let output_pred = NamedNode::new(format!("{}output", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(output_term) = self.get_object(&subject, &output_pred) {
            let output_urn = self.term_to_string(&output_term)?;
            match self.parse_property(&output_urn) {
                Ok(property) => {
                    operation.output = Some(property);
                }
                Err(e) => {
                    tracing::warn!("Failed to parse output property '{}': {}", output_urn, e);
                }
            }
        }

        Ok(operation)
    }

    /// Parse an Event from the graph
    fn parse_event(&mut self, urn: &str) -> Result<Event> {
        let subject = NamedNode::new(urn)
            .map_err(|e| SammError::ParseError(format!("Invalid URN '{}': {}", urn, e)))?;

        let mut event = Event::new(urn.to_string());

        // Parse metadata
        self.parse_element_metadata(&subject, &mut event.metadata)?;

        // Parse parameters (RDF list)
        let parameters_pred =
            NamedNode::new(format!("{}parameters", self.samm_ns.as_ref().unwrap()))
                .map_err(|e| SammError::ParseError(e.to_string()))?;

        if let Some(parameters_term) = self.get_object(&subject, &parameters_pred) {
            let parameter_urns = self.parse_rdf_list(&parameters_term)?;
            for param_urn in parameter_urns {
                match self.parse_property(&param_urn) {
                    Ok(property) => event.add_parameter(property),
                    Err(e) => {
                        tracing::warn!("Failed to parse event parameter '{}': {}", param_urn, e);
                    }
                }
            }
        }

        Ok(event)
    }

    /// Parse element metadata (preferredName, description, see)
    fn parse_element_metadata(
        &self,
        subject: &NamedNode,
        metadata: &mut ElementMetadata,
    ) -> Result<()> {
        // Parse preferredName
        let pref_name_pred =
            NamedNode::new(format!("{}preferredName", self.samm_ns.as_ref().unwrap()))
                .map_err(|e| SammError::ParseError(e.to_string()))?;

        for triple in self.graph.iter().filter(|t| {
            if let NamedOrBlankNodeRef::NamedNode(s) = t.subject {
                s == subject.as_ref() && t.predicate == pref_name_pred.as_ref()
            } else {
                false
            }
        }) {
            if let TermRef::Literal(lit) = triple.object {
                let lang = lit.language().unwrap_or("en");
                metadata.add_preferred_name(lang.to_string(), lit.value().to_string());
            }
        }

        // Parse description
        let desc_pred = NamedNode::new(format!("{}description", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        for triple in self.graph.iter().filter(|t| {
            if let NamedOrBlankNodeRef::NamedNode(s) = t.subject {
                s == subject.as_ref() && t.predicate == desc_pred.as_ref()
            } else {
                false
            }
        }) {
            if let TermRef::Literal(lit) = triple.object {
                let lang = lit.language().unwrap_or("en");
                metadata.add_description(lang.to_string(), lit.value().to_string());
            }
        }

        // Parse see
        let see_pred = NamedNode::new(format!("{}see", self.samm_ns.as_ref().unwrap()))
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        for triple in self.graph.iter().filter(|t| {
            if let NamedOrBlankNodeRef::NamedNode(s) = t.subject {
                s == subject.as_ref() && t.predicate == see_pred.as_ref()
            } else {
                false
            }
        }) {
            if let TermRef::NamedNode(node) = triple.object {
                metadata.add_see_ref(node.as_str().to_string());
            }
        }

        Ok(())
    }

    /// Get the object of a triple for a given subject and predicate
    fn get_object(&self, subject: &NamedNode, predicate: &NamedNode) -> Option<Term> {
        self.graph
            .iter()
            .find(|triple| {
                if let NamedOrBlankNodeRef::NamedNode(s) = triple.subject {
                    s == subject.as_ref() && triple.predicate == predicate.as_ref()
                } else {
                    false
                }
            })
            .map(|triple| triple.object.into_owned())
    }

    /// Convert a Term to a String
    fn term_to_string(&self, term: &Term) -> Result<String> {
        match term {
            Term::NamedNode(node) => Ok(node.as_str().to_string()),
            Term::BlankNode(bnode) => Ok(format!("_:{}", bnode.as_str())),
            Term::Literal(lit) => Ok(lit.value().to_string()),
            #[allow(unreachable_patterns)]
            _ => Err(SammError::ParseError("Unsupported term type".to_string())),
        }
    }

    /// Parse an RDF list into a vector of URNs
    fn parse_rdf_list(&self, list_term: &Term) -> Result<Vec<String>> {
        let mut result = Vec::new();
        let mut current = list_term.clone();

        let rdf_first = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first")
            .map_err(|e| SammError::ParseError(e.to_string()))?;
        let rdf_rest = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest")
            .map_err(|e| SammError::ParseError(e.to_string()))?;
        let rdf_nil = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil")
            .map_err(|e| SammError::ParseError(e.to_string()))?;

        loop {
            match &current {
                Term::NamedNode(node) if node == &rdf_nil => break,
                Term::NamedNode(_) | Term::BlankNode(_) => {
                    // Find the rdf:first triple for this subject
                    let first_obj = self.graph.iter().find_map(|triple| {
                        let subject_matches = match (&current, triple.subject) {
                            (Term::NamedNode(n), NamedOrBlankNodeRef::NamedNode(s)) => {
                                n.as_ref() == s
                            }
                            (Term::BlankNode(b), NamedOrBlankNodeRef::BlankNode(s)) => {
                                b.as_ref() == s
                            }
                            _ => false,
                        };

                        if subject_matches && triple.predicate == rdf_first.as_ref() {
                            Some(triple.object.into_owned())
                        } else {
                            None
                        }
                    });

                    if let Some(first) = first_obj {
                        result.push(self.term_to_string(&first)?);
                    }

                    // Find the rdf:rest triple for this subject
                    let rest_obj = self.graph.iter().find_map(|triple| {
                        let subject_matches = match (&current, triple.subject) {
                            (Term::NamedNode(n), NamedOrBlankNodeRef::NamedNode(s)) => {
                                n.as_ref() == s
                            }
                            (Term::BlankNode(b), NamedOrBlankNodeRef::BlankNode(s)) => {
                                b.as_ref() == s
                            }
                            _ => false,
                        };

                        if subject_matches && triple.predicate == rdf_rest.as_ref() {
                            Some(triple.object.into_owned())
                        } else {
                            None
                        }
                    });

                    if let Some(rest) = rest_obj {
                        current = rest;
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        Ok(result)
    }
}

impl Default for SammTurtleParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parser_creation() {
        let parser = SammTurtleParser::new();
        assert_eq!(parser.graph.len(), 0);
    }

    #[tokio::test]
    async fn test_parse_simple_aspect() {
        let ttl = r#"
@prefix : <urn:samm:com.example:1.0.0#> .
@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:TestAspect a samm:Aspect ;
   samm:preferredName "Test Aspect"@en ;
   samm:description "A test aspect"@en ;
   samm:properties ( ) ;
   samm:operations ( ) ;
   samm:events ( ) .
"#;

        let mut parser = SammTurtleParser::new();
        let result = parser.parse_string(ttl, "http://example.org").await;

        assert!(result.is_ok(), "Parse failed: {:?}", result.err());
        let aspect = result.unwrap();
        assert_eq!(aspect.name(), "TestAspect");
        assert_eq!(
            aspect.metadata.get_preferred_name("en"),
            Some("Test Aspect")
        );
    }
}
