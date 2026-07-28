use std::collections::HashMap;

use tracing::{debug, span, Level};

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::StarResult;

use super::types::{
    AdvancedReificationStrategy, ReificationCondition, ReificationContext, ReificationStatistics,
    ReificationStrategy, TermType,
};
use super::vocab;

/// Partially-discovered `rdf:subject`/`rdf:predicate`/`rdf:object` components
/// for a candidate statement identifier during dereification, in
/// (subject, predicate, object) order.
type PartialStatementComponents = (Option<StarTerm>, Option<StarTerm>, Option<StarTerm>);

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
                StarTerm::iri(vocab::RDF_SINGLETON_PROPERTY_OF)?,
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

    /// Dereify a standard-RDF graph back into RDF-star, reversing whichever
    /// [`ReificationStrategy`] was used to produce it.
    ///
    /// This is strategy-agnostic: it does not rely on the presence of the
    /// optional `rdf:type rdf:Statement` marker triple, since `UniqueIris`
    /// and `BlankNodes` never emit one, and `SingletonProperties` uses a
    /// completely different shape (`s singletonProp o` + `singletonProp
    /// rdf:singletonPropertyOf p`). Instead a statement identifier is
    /// recognized structurally:
    ///   * standard/unique-iri/blank-node reification: a term (NamedNode or
    ///     BlankNode) that has a complete `rdf:subject`/`rdf:predicate`/
    ///     `rdf:object` triple triple (the `rdf:type rdf:Statement` marker,
    ///     if present, is treated as optional confirmation, not a
    ///     requirement).
    ///   * singleton properties: a NamedNode `p1` with a
    ///     `p1 rdf:singletonPropertyOf p` triple and a matching `s p1 o`
    ///     data triple.
    ///
    /// Once every statement identifier has been resolved to its
    /// reconstructed `StarTriple`, every remaining (non-scaffolding) triple
    /// has each of its subject/predicate/object positions substituted with
    /// the reconstructed `<<s p o>>` quoted triple wherever that position
    /// refers to a resolved statement identifier -- not just the subject
    /// position.
    pub fn dereify_graph(&mut self, reified_graph: &StarGraph) -> StarResult<StarGraph> {
        let span = span!(Level::INFO, "dereify_graph");
        let _enter = span.enter();

        let mut star_graph = StarGraph::new();

        // --- Pass 1: discover statement identifiers -----------------------

        // Partial rdf:subject/rdf:predicate/rdf:object accumulation, keyed
        // by the candidate statement identifier term (NamedNode or
        // BlankNode subject of those triples).
        let mut partial: HashMap<StarTerm, PartialStatementComponents> = HashMap::new();
        // Singleton-property identifier -> original predicate it stands in for.
        let mut singleton_props: HashMap<StarTerm, StarTerm> = HashMap::new();

        for triple in reified_graph.triples() {
            if !matches!(
                triple.subject,
                StarTerm::NamedNode(_) | StarTerm::BlankNode(_)
            ) {
                continue;
            }
            if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                match pred_node.iri.as_str() {
                    vocab::RDF_SUBJECT => {
                        partial.entry(triple.subject.clone()).or_default().0 =
                            Some(triple.object.clone());
                    }
                    vocab::RDF_PREDICATE => {
                        partial.entry(triple.subject.clone()).or_default().1 =
                            Some(triple.object.clone());
                    }
                    vocab::RDF_OBJECT => {
                        partial.entry(triple.subject.clone()).or_default().2 =
                            Some(triple.object.clone());
                    }
                    vocab::RDF_SINGLETON_PROPERTY_OF => {
                        singleton_props.insert(triple.subject.clone(), triple.object.clone());
                    }
                    _ => {}
                }
            }
        }

        // Resolve standard/unique-iri/blank-node style statement identifiers:
        // only complete (subject+predicate+object all present) entries count.
        let mut reconstructed: HashMap<StarTerm, StarTriple> = HashMap::new();
        for (stmt_term, (subject, predicate, object)) in partial {
            if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                reconstructed.insert(stmt_term, StarTriple::new(s, p, o));
            }
        }

        // Resolve singleton-property style statement identifiers: find the
        // data triple `s property o` for each known `property`.
        if !singleton_props.is_empty() {
            for triple in reified_graph.triples() {
                if reconstructed.contains_key(&triple.predicate) {
                    // Already resolved by another pass or a duplicate data
                    // triple for the same property; keep first match.
                    continue;
                }
                if let Some(orig_predicate) = singleton_props.get(&triple.predicate) {
                    reconstructed.insert(
                        triple.predicate.clone(),
                        StarTriple::new(
                            triple.subject.clone(),
                            orig_predicate.clone(),
                            triple.object.clone(),
                        ),
                    );
                }
            }
        }

        // --- Pass 2: emit non-scaffolding triples with substitution -------

        for triple in reified_graph.triples() {
            if self.is_reification_meta_triple(triple, &reconstructed, &singleton_props) {
                continue;
            }

            let subject = Self::substitute_statement_ref(&triple.subject, &reconstructed);
            let predicate = Self::substitute_statement_ref(&triple.predicate, &reconstructed);
            let object = Self::substitute_statement_ref(&triple.object, &reconstructed);

            star_graph.insert(StarTriple::new(subject, predicate, object))?;
        }

        debug!(
            "Dereified {} reified triples back to {} RDF-star triples",
            reified_graph.len(),
            star_graph.len()
        );
        Ok(star_graph)
    }

    /// Replace `term` with the reconstructed quoted triple if it refers to a
    /// resolved statement identifier; otherwise return it unchanged. Used to
    /// substitute references appearing in *any* of the subject, predicate,
    /// or object positions.
    fn substitute_statement_ref(
        term: &StarTerm,
        reconstructed: &HashMap<StarTerm, StarTriple>,
    ) -> StarTerm {
        match reconstructed.get(term) {
            Some(quoted) => StarTerm::quoted_triple(quoted.clone()),
            None => term.clone(),
        }
    }

    /// Determine whether `triple` is scaffolding produced by
    /// [`Self::create_reification_triples`] for one of the statement
    /// identifiers in `reconstructed`, and therefore must not be re-emitted
    /// verbatim into the dereified graph (it is fully represented by the
    /// reconstructed quoted triple substituted elsewhere).
    fn is_reification_meta_triple(
        &self,
        triple: &StarTriple,
        reconstructed: &HashMap<StarTerm, StarTriple>,
        singleton_props: &HashMap<StarTerm, StarTerm>,
    ) -> bool {
        // rdf:type / rdf:subject / rdf:predicate / rdf:object scaffolding,
        // keyed by a resolved statement identifier in subject position.
        if reconstructed.contains_key(&triple.subject) {
            if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                match pred_node.iri.as_str() {
                    vocab::RDF_TYPE
                    | vocab::RDF_SUBJECT
                    | vocab::RDF_PREDICATE
                    | vocab::RDF_OBJECT => return true,
                    _ => {}
                }
            }
        }

        // Singleton-property scaffolding: `property rdf:singletonPropertyOf
        // origPredicate`.
        if singleton_props.contains_key(&triple.subject) {
            if let StarTerm::NamedNode(pred_node) = &triple.predicate {
                if pred_node.iri == vocab::RDF_SINGLETON_PROPERTY_OF {
                    return true;
                }
            }
        }

        // Singleton-property scaffolding: the `s property o` data triple
        // whose predicate *is* the resolved statement identifier.
        if singleton_props.contains_key(&triple.predicate)
            && reconstructed.contains_key(&triple.predicate)
        {
            return true;
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
