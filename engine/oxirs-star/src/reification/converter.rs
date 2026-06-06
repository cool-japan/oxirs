use std::collections::HashMap;

use tracing::{debug, span, Level};

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::StarResult;

use super::types::{
    AdvancedReificationStrategy, ReificationCondition, ReificationContext, ReificationStatistics,
    ReificationStrategy, TermType,
};
use super::vocab;

/// RDF-star to standard RDF reification converter
pub struct Reificator {
    pub context: ReificationContext,
}

impl Reificator {
    pub fn new(strategy: ReificationStrategy, base_iri: Option<String>) -> Self {
        Self {
            context: ReificationContext::new(strategy, base_iri),
        }
    }

    pub fn reify_graph(&mut self, star_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "reify_graph");
        let _enter = span.enter();

        let mut reified_graph = StarGraph::new();

        for triple in star_graph.triples() {
            let reified_triples = self.reify_triple(triple)?;
            for reified_triple in reified_triples {
                reified_graph.insert(reified_triple)?;
            }
        }

        debug!(
            "Reified {} triples to {} standard RDF triples",
            star_graph.len(),
            reified_graph.len()
        );
        Ok(reified_graph)
    }

    pub fn reify_triple(&mut self, triple: &StarTriple) -> StarResult<Vec<StarTriple>> {
        let mut result = Vec::new();

        let subject = self.reify_term(&triple.subject, &mut result)?;
        let predicate = self.reify_term(&triple.predicate, &mut result)?;
        let object = self.reify_term(&triple.object, &mut result)?;

        let main_triple = StarTriple::new(subject, predicate, object);
        result.push(main_triple);

        Ok(result)
    }

    fn reify_term(
        &mut self,
        term: &StarTerm,
        additional_triples: &mut Vec<StarTriple>,
    ) -> StarResult<StarTerm> {
        match term {
            StarTerm::QuotedTriple(quoted_triple) => {
                let stmt_id = self.context.generate_id(quoted_triple);
                let reification_triples =
                    self.create_reification_triples(&stmt_id, quoted_triple)?;
                additional_triples.extend(reification_triples);

                match self.context.strategy {
                    ReificationStrategy::StandardReification | ReificationStrategy::UniqueIris => {
                        Ok(StarTerm::iri(&stmt_id)?)
                    }
                    ReificationStrategy::BlankNodes => {
                        let blank_id = &stmt_id[2..];
                        Ok(StarTerm::blank_node(blank_id)?)
                    }
                    ReificationStrategy::SingletonProperties => Ok(StarTerm::iri(&stmt_id)?),
                }
            }
            _ => Ok(term.clone()),
        }
    }

    fn create_reification_triples(
        &mut self,
        stmt_id: &str,
        triple: &StarTriple,
    ) -> StarResult<Vec<StarTriple>> {
        let mut triples = Vec::new();

        if matches!(
            self.context.strategy,
            ReificationStrategy::SingletonProperties
        ) {
            let property_term = StarTerm::iri(stmt_id)?;

            let mut subject_additional = Vec::new();
            let reified_subject = self.reify_term(&triple.subject, &mut subject_additional)?;
            triples.extend(subject_additional);

            let mut object_additional = Vec::new();
            let reified_object = self.reify_term(&triple.object, &mut object_additional)?;
            triples.extend(object_additional);

            triples.push(StarTriple::new(
                reified_subject,
                property_term.clone(),
                reified_object,
            ));

            triples.push(StarTriple::new(
                property_term,
                StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#singletonPropertyOf")?,
                triple.predicate.clone(),
            ));

            return Ok(triples);
        }

        let stmt_term = match self.context.strategy {
            ReificationStrategy::StandardReification | ReificationStrategy::UniqueIris => {
                StarTerm::iri(stmt_id)?
            }
            ReificationStrategy::BlankNodes => {
                let blank_id = &stmt_id[2..];
                StarTerm::blank_node(blank_id)?
            }
            ReificationStrategy::SingletonProperties => {
                unreachable!("Handled above")
            }
        };

        if matches!(
            self.context.strategy,
            ReificationStrategy::StandardReification
        ) {
            triples.push(StarTriple::new(
                stmt_term.clone(),
                StarTerm::iri(vocab::RDF_TYPE)?,
                StarTerm::iri(vocab::RDF_STATEMENT)?,
            ));
        }

        let mut subject_additional = Vec::new();
        let reified_subject = self.reify_term(&triple.subject, &mut subject_additional)?;
        triples.extend(subject_additional);

        let mut predicate_additional = Vec::new();
        let reified_predicate = self.reify_term(&triple.predicate, &mut predicate_additional)?;
        triples.extend(predicate_additional);

        let mut object_additional = Vec::new();
        let reified_object = self.reify_term(&triple.object, &mut object_additional)?;
        triples.extend(object_additional);

        triples.push(StarTriple::new(
            stmt_term.clone(),
            StarTerm::iri(vocab::RDF_SUBJECT)?,
            reified_subject,
        ));

        triples.push(StarTriple::new(
            stmt_term.clone(),
            StarTerm::iri(vocab::RDF_PREDICATE)?,
            reified_predicate,
        ));

        triples.push(StarTriple::new(
            stmt_term,
            StarTerm::iri(vocab::RDF_OBJECT)?,
            reified_object,
        ));

        Ok(triples)
    }

    pub fn dereify_graph(&mut self, reified_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "dereify_graph");
        let _enter = span.enter();

        let mut star_graph = StarGraph::new();
        let mut processed_statements = std::collections::HashSet::new();
        let mut reconstructed_triples = std::collections::HashMap::new();

        for triple in reified_graph.triples() {
            if let (StarTerm::NamedNode(predicate), StarTerm::NamedNode(object)) =
                (&triple.predicate, &triple.object)
            {
                if predicate.iri == vocab::RDF_TYPE && object.iri == vocab::RDF_STATEMENT {
                    if let StarTerm::NamedNode(stmt_node) = &triple.subject {
                        if !processed_statements.contains(&stmt_node.iri) {
                            if let Some(star_triple) =
                                self.reconstruct_quoted_triple(reified_graph, &stmt_node.iri)?
                            {
                                reconstructed_triples.insert(stmt_node.iri.clone(), star_triple);
                                processed_statements.insert(stmt_node.iri.clone());
                            }
                        }
                    }
                }
            }
        }

        for triple in reified_graph.triples() {
            if self.is_reification_meta_triple(triple, &processed_statements) {
                continue;
            }

            if let StarTerm::NamedNode(subject_node) = &triple.subject {
                if let Some(quoted_triple) = reconstructed_triples.get(&subject_node.iri) {
                    let new_triple = StarTriple::new(
                        StarTerm::quoted_triple(quoted_triple.clone()),
                        triple.predicate.clone(),
                        triple.object.clone(),
                    );
                    star_graph.insert(new_triple)?;
                } else {
                    star_graph.insert(triple.clone())?;
                }
            } else {
                star_graph.insert(triple.clone())?;
            }
        }

        debug!(
            "Dereified {} reified triples back to {} RDF-star triples",
            reified_graph.len(),
            star_graph.len()
        );
        Ok(star_graph)
    }

    fn reconstruct_quoted_triple(
        &self,
        graph: &StarGraph,
        stmt_iri: &str,
    ) -> StarResult<Option<StarTriple>> {
        let mut subject = None;
        let mut predicate = None;
        let mut object = None;

        let stmt_term = StarTerm::iri(stmt_iri)?;

        for triple in graph.triples() {
            if triple.subject == stmt_term {
                if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                    match pred_node.iri.as_str() {
                        vocab::RDF_SUBJECT => subject = Some(triple.object.clone()),
                        vocab::RDF_PREDICATE => predicate = Some(triple.object.clone()),
                        vocab::RDF_OBJECT => object = Some(triple.object.clone()),
                        _ => {}
                    }
                }
            }
        }

        if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
            Ok(Some(StarTriple::new(s, p, o)))
        } else {
            Ok(None)
        }
    }

    fn is_reification_meta_triple(
        &self,
        triple: &StarTriple,
        processed_statements: &std::collections::HashSet<String>,
    ) -> bool {
        if let StarTerm::NamedNode(subj_node) = &triple.subject {
            if processed_statements.contains(&subj_node.iri) {
                if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                    match pred_node.iri.as_str() {
                        vocab::RDF_TYPE
                        | vocab::RDF_SUBJECT
                        | vocab::RDF_PREDICATE
                        | vocab::RDF_OBJECT => {
                            return true;
                        }
                        _ => {}
                    }
                }
            }
        }
        false
    }
}

/// Enhanced reificator with advanced strategies
pub struct AdvancedReificator {
    strategy: AdvancedReificationStrategy,
    contexts: HashMap<String, ReificationContext>,
    #[allow(dead_code)]
    cache: lru::LruCache<String, Vec<StarTriple>>,
    statistics: ReificationStatistics,
}

impl AdvancedReificator {
    pub fn new(strategy: AdvancedReificationStrategy) -> Self {
        Self {
            strategy,
            contexts: HashMap::new(),
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(1000).expect("1000 is non-zero")),
            statistics: ReificationStatistics::default(),
        }
    }

    pub fn reify_graph_advanced(&mut self, star_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "reify_graph_advanced");
        let _enter = span.enter();

        let start_time = std::time::Instant::now();
        let mut reified_graph = StarGraph::new();

        for triple in star_graph.triples() {
            self.statistics.total_triples += 1;

            let strategy = self.select_strategy_for_triple(triple)?;
            let strategy_name = format!("{strategy:?}");
            *self
                .statistics
                .strategy_usage
                .entry(strategy_name)
                .or_insert(0) += 1;

            let reified_triples = self.reify_triple_with_strategy(triple, &strategy)?;
            for reified_triple in reified_triples {
                reified_graph.insert(reified_triple)?;
                self.statistics.reification_triples += 1;
            }
        }

        let processing_time = start_time.elapsed();
        self.statistics.avg_processing_time =
            processing_time.as_micros() as f64 / self.statistics.total_triples as f64;

        debug!(
            "Advanced reification completed: {} triples -> {} triples in {:?}",
            star_graph.len(),
            reified_graph.len(),
            processing_time
        );

        Ok(reified_graph)
    }

    fn select_strategy_for_triple(&self, triple: &StarTriple) -> StarResult<ReificationStrategy> {
        match &self.strategy {
            AdvancedReificationStrategy::Standard(strategy) => Ok(strategy.clone()),
            AdvancedReificationStrategy::Hybrid {
                simple_strategy,
                nested_strategy,
                predicate_strategies,
            } => {
                if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                    if let Some(strategy) = predicate_strategies.get(&pred_node.iri) {
                        return Ok(strategy.clone());
                    }
                }

                if self.has_nested_quoted_triples(triple) {
                    Ok(nested_strategy.clone())
                } else {
                    Ok(simple_strategy.clone())
                }
            }
            AdvancedReificationStrategy::Conditional {
                default_strategy,
                rules,
            } => {
                let mut applicable_rules: Vec<_> = rules
                    .iter()
                    .filter(|rule| self.evaluate_condition(&rule.condition, triple))
                    .collect();
                applicable_rules.sort_by_key(|rule| std::cmp::Reverse(rule.priority));

                if let Some(rule) = applicable_rules.first() {
                    Ok(rule.strategy.clone())
                } else {
                    Ok(default_strategy.clone())
                }
            }
            AdvancedReificationStrategy::Optimized { base_strategy, .. } => {
                Ok(base_strategy.clone())
            }
        }
    }

    fn has_nested_quoted_triples(&self, triple: &StarTriple) -> bool {
        self.term_has_quoted_triples(&triple.subject)
            || self.term_has_quoted_triples(&triple.predicate)
            || self.term_has_quoted_triples(&triple.object)
    }

    fn term_has_quoted_triples(&self, term: &StarTerm) -> bool {
        matches!(term, StarTerm::QuotedTriple(_))
    }

    fn evaluate_condition(&self, condition: &ReificationCondition, triple: &StarTriple) -> bool {
        match condition {
            ReificationCondition::PredicateIri(iri) => {
                if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                    pred_node.iri == *iri
                } else {
                    false
                }
            }
            ReificationCondition::SubjectType(term_type) => {
                self.matches_term_type(&triple.subject, term_type)
            }
            ReificationCondition::ObjectType(term_type) => {
                self.matches_term_type(&triple.object, term_type)
            }
            ReificationCondition::NestingDepth(max_depth) => {
                self.calculate_nesting_depth(triple) <= *max_depth
            }
            ReificationCondition::GraphSize(_) => true,
            ReificationCondition::Custom(_) => false,
        }
    }

    fn matches_term_type(&self, term: &StarTerm, term_type: &TermType) -> bool {
        matches!(
            (term, term_type),
            (StarTerm::NamedNode(_), TermType::NamedNode)
                | (StarTerm::BlankNode(_), TermType::BlankNode)
                | (StarTerm::Literal(_), TermType::Literal)
                | (StarTerm::QuotedTriple(_), TermType::QuotedTriple)
                | (StarTerm::Variable(_), TermType::Variable)
        )
    }

    fn calculate_nesting_depth(&self, triple: &StarTriple) -> usize {
        let subject_depth = self.term_nesting_depth(&triple.subject);
        let predicate_depth = self.term_nesting_depth(&triple.predicate);
        let object_depth = self.term_nesting_depth(&triple.object);

        subject_depth.max(predicate_depth).max(object_depth)
    }

    fn term_nesting_depth(&self, term: &StarTerm) -> usize {
        match term {
            StarTerm::QuotedTriple(inner_triple) => 1 + self.calculate_nesting_depth(inner_triple),
            _ => 0,
        }
    }

    fn reify_triple_with_strategy(
        &mut self,
        triple: &StarTriple,
        strategy: &ReificationStrategy,
    ) -> StarResult<Vec<StarTriple>> {
        let context_key = format!("{strategy:?}");
        if !self.contexts.contains_key(&context_key) {
            self.contexts.insert(
                context_key.clone(),
                ReificationContext::new(strategy.clone(), None),
            );
        }

        let context = self
            .contexts
            .get_mut(&context_key)
            .expect("context should exist after insertion");
        let mut temp_reificator = Reificator {
            context: ReificationContext::new(strategy.clone(), None),
        };

        temp_reificator.context.counter = context.counter;
        temp_reificator.context.triple_to_id = context.triple_to_id.clone();
        temp_reificator.context.id_to_triple = context.id_to_triple.clone();

        let result = temp_reificator.reify_triple(triple);

        context.counter = temp_reificator.context.counter;
        context.triple_to_id = temp_reificator.context.triple_to_id;
        context.id_to_triple = temp_reificator.context.id_to_triple;

        result
    }

    pub fn get_statistics(&self) -> &ReificationStatistics {
        &self.statistics
    }

    pub fn reset_statistics(&mut self) {
        self.statistics = ReificationStatistics::default();
    }

    pub fn export_mappings(&self) -> HashMap<String, HashMap<String, String>> {
        let mut mappings = HashMap::new();

        for (strategy_key, context) in &self.contexts {
            mappings.insert(strategy_key.clone(), context.triple_to_id.clone());
        }

        mappings
    }
}
