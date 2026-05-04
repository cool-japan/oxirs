# SHACL Specification → OxiRS Symbol Mapping

This document is the cross-reference table between the W3C **SHACL** /
**SHACL-AF** specifications and the Rust symbols in `oxirs_shacl` that
implement them. Use it when you need to know "where does *X* live in the
codebase?".

**Sources of truth.**

- [SHACL Core](https://www.w3.org/TR/shacl/) (W3C Recommendation, 20 July 2017)
- [SHACL Advanced Features](https://www.w3.org/TR/shacl-af/) (W3C Working Group Note, 8 June 2017)

The "Symbol" column lists the canonical struct or function. Anything
qualifying it (`Constraint::*` enum variant, `ConstraintComponentId`, etc.) is
linked from the row's notes.

> **Convention.** Every constraint reachable from a shapes graph passes through
> the [`Constraint`] enum in `engine/oxirs-shacl/src/constraints/constraint_types.rs`.
> The variant's `component_id()` returns the constraint component IRI shown in
> validation reports.

[`Constraint`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/constraint_types/enum.Constraint.html

---

## 1. Top-level vocabulary

| Concept                      | Spec section           | Symbol                                              |
|------------------------------|------------------------|-----------------------------------------------------|
| `sh:NodeShape`               | Core §2.2              | [`Shape`] with [`ShapeType::NodeShape`]             |
| `sh:PropertyShape`           | Core §2.3              | [`Shape`] with [`ShapeType::PropertyShape`]         |
| `sh:Shape` (abstract)        | Core §2.1              | [`Shape`]                                           |
| `sh:deactivated`             | Core §2.1.6            | `Shape::deactivated` field, `Shape::is_active()`    |
| `sh:severity`                | Core §2.1.4            | [`Severity`] (`Info` / `Warning` / `Violation`)    |
| `sh:message`                 | Core §2.1.5            | `Shape::messages`, `PatternConstraint::message`, `SparqlConstraint::message` |
| `sh:order`                   | Core §2.3.2.2          | `Shape::order` field                                |
| `sh:group` / `sh:PropertyGroup` | Core §2.3.2.3       | `Shape::groups` field                               |
| `sh:targetNode`              | Core §2.1.3.1          | [`Target::Node`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.Node) |
| `sh:targetClass`             | Core §2.1.3.2          | [`Target::Class`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.Class) |
| Implicit class target        | Core §2.1.3.3          | [`Target::Implicit`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.Implicit) |
| `sh:targetSubjectsOf`        | Core §2.1.3.4          | [`Target::SubjectsOf`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.SubjectsOf) |
| `sh:targetObjectsOf`         | Core §2.1.3.5          | [`Target::ObjectsOf`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.ObjectsOf) |
| Validation report            | Core §3                | [`ValidationReport`]                                |
| Validation result            | Core §3.4              | [`ValidationViolation`]                             |

[`Shape`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/struct.Shape.html
[`ShapeType::NodeShape`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/enum.ShapeType.html#variant.NodeShape
[`ShapeType::PropertyShape`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/enum.ShapeType.html#variant.PropertyShape
[`Severity`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/enum.Severity.html
[`ValidationReport`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/report/struct.ValidationReport.html
[`ValidationViolation`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/validation/struct.ValidationViolation.html

---

## 2. Property paths (`sh:path`)

| Path operator                | Spec section | Symbol                                             |
|------------------------------|--------------|----------------------------------------------------|
| Predicate path               | Core §2.3.1.1 | [`PropertyPath::Predicate`]                       |
| Sequence path                | Core §2.3.1.2 | [`PropertyPath::Sequence`]                        |
| Alternative path             | Core §2.3.1.3 | [`PropertyPath::Alternative`]                     |
| Inverse path                 | Core §2.3.1.4 | [`PropertyPath::Inverse`]                         |
| Zero-or-more path            | Core §2.3.1.5 | [`PropertyPath::ZeroOrMore`]                      |
| One-or-more path             | Core §2.3.1.6 | [`PropertyPath::OneOrMore`]                       |
| Zero-or-one path             | Core §2.3.1.7 | [`PropertyPath::ZeroOrOne`]                       |
| Path evaluator (cache + SPARQL fallback) | (impl) | `oxirs_shacl::paths::PropertyPathEvaluator` |
| Path execution / traversal   | (impl)       | `oxirs_shacl::path_executor`                       |
| Path checker / structural validation | (impl) | `oxirs_shacl::property_path_checker`             |

[`PropertyPath::Predicate`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/paths/types/enum.PropertyPath.html#variant.Predicate
[`PropertyPath::Inverse`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/paths/types/enum.PropertyPath.html#variant.Inverse
[`PropertyPath::Sequence`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/paths/types/enum.PropertyPath.html#variant.Sequence
[`PropertyPath::Alternative`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/paths/types/enum.PropertyPath.html#variant.Alternative
[`PropertyPath::ZeroOrMore`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/paths/types/enum.PropertyPath.html#variant.ZeroOrMore
[`PropertyPath::OneOrMore`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/paths/types/enum.PropertyPath.html#variant.OneOrMore
[`PropertyPath::ZeroOrOne`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/paths/types/enum.PropertyPath.html#variant.ZeroOrOne

---

## 3. SHACL Core constraint components

The 27 constraint components defined in SHACL Core §4. Each row gives the
SHACL parameter, the spec section, the [`Constraint`] enum variant, and the
backing struct that implements `evaluate()`.

| SHACL parameter             | Component IRI                                  | Spec   | `Constraint` variant | Struct                                                                                       |
|-----------------------------|-------------------------------------------------|--------|----------------------|-----------------------------------------------------------------------------------------------|
| `sh:class`                  | `sh:ClassConstraintComponent`                   | §4.1.1 | `Class`              | [`ClassConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/value_constraints/struct.ClassConstraint.html) |
| `sh:datatype`               | `sh:DatatypeConstraintComponent`                | §4.1.2 | `Datatype`           | [`DatatypeConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/value_constraints/struct.DatatypeConstraint.html) |
| `sh:nodeKind`               | `sh:NodeKindConstraintComponent`                | §4.1.3 | `NodeKind`           | [`NodeKindConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/value_constraints/struct.NodeKindConstraint.html) |
| `sh:minCount`               | `sh:MinCountConstraintComponent`                | §4.2.1 | `MinCount`           | [`MinCountConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/cardinality_constraints/struct.MinCountConstraint.html) |
| `sh:maxCount`               | `sh:MaxCountConstraintComponent`                | §4.2.2 | `MaxCount`           | [`MaxCountConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/cardinality_constraints/struct.MaxCountConstraint.html) |
| `sh:minExclusive`           | `sh:MinExclusiveConstraintComponent`            | §4.3.1 | `MinExclusive`       | [`MinExclusiveConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/range_constraints/struct.MinExclusiveConstraint.html) |
| `sh:minInclusive`           | `sh:MinInclusiveConstraintComponent`            | §4.3.2 | `MinInclusive`       | [`MinInclusiveConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/range_constraints/struct.MinInclusiveConstraint.html) |
| `sh:maxExclusive`           | `sh:MaxExclusiveConstraintComponent`            | §4.3.3 | `MaxExclusive`       | [`MaxExclusiveConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/range_constraints/struct.MaxExclusiveConstraint.html) |
| `sh:maxInclusive`           | `sh:MaxInclusiveConstraintComponent`            | §4.3.4 | `MaxInclusive`       | [`MaxInclusiveConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/range_constraints/struct.MaxInclusiveConstraint.html) |
| `sh:minLength`              | `sh:MinLengthConstraintComponent`               | §4.4.1 | `MinLength`          | [`MinLengthConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/string_constraints/struct.MinLengthConstraint.html) |
| `sh:maxLength`              | `sh:MaxLengthConstraintComponent`               | §4.4.2 | `MaxLength`          | [`MaxLengthConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/string_constraints/struct.MaxLengthConstraint.html) |
| `sh:pattern` / `sh:flags`   | `sh:PatternConstraintComponent`                 | §4.4.3 | `Pattern`            | [`PatternConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/string_constraints/struct.PatternConstraint.html) |
| `sh:languageIn`             | `sh:LanguageInConstraintComponent`              | §4.4.4 | `LanguageIn`         | [`LanguageInConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/string_constraints/struct.LanguageInConstraint.html) |
| `sh:uniqueLang`             | `sh:UniqueLangConstraintComponent`              | §4.4.5 | `UniqueLang`         | [`UniqueLangConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/string_constraints/struct.UniqueLangConstraint.html) |
| `sh:equals`                 | `sh:EqualsConstraintComponent`                  | §4.5.1 | `Equals`             | [`EqualsConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/comparison_constraints/struct.EqualsConstraint.html) |
| `sh:disjoint`               | `sh:DisjointConstraintComponent`                | §4.5.2 | `Disjoint`           | [`DisjointConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/comparison_constraints/struct.DisjointConstraint.html) |
| `sh:lessThan`               | `sh:LessThanConstraintComponent`                | §4.5.3 | `LessThan`           | [`LessThanConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/comparison_constraints/struct.LessThanConstraint.html) |
| `sh:lessThanOrEquals`       | `sh:LessThanOrEqualsConstraintComponent`        | §4.5.4 | `LessThanOrEquals`   | [`LessThanOrEqualsConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/comparison_constraints/struct.LessThanOrEqualsConstraint.html) |
| `sh:not`                    | `sh:NotConstraintComponent`                     | §4.6.1 | `Not`                | [`NotConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/logical_constraints/struct.NotConstraint.html) |
| `sh:and`                    | `sh:AndConstraintComponent`                     | §4.6.2 | `And`                | [`AndConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/logical_constraints/struct.AndConstraint.html) |
| `sh:or`                     | `sh:OrConstraintComponent`                      | §4.6.3 | `Or`                 | [`OrConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/logical_constraints/struct.OrConstraint.html) |
| `sh:xone`                   | `sh:XoneConstraintComponent`                    | §4.6.4 | `Xone`               | [`XoneConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/logical_constraints/struct.XoneConstraint.html) |
| `sh:node`                   | `sh:NodeConstraintComponent`                    | §4.7.1 | `Node`               | [`NodeConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/shape_constraints/struct.NodeConstraint.html) |
| `sh:property`               | `sh:PropertyConstraintComponent`                | §4.7.2 | (handled via [`Shape::property_shapes`])     | [`PropertyConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/shape_constraints/struct.PropertyConstraint.html) |
| `sh:qualifiedValueShape` / `sh:qualifiedMinCount` / `sh:qualifiedMaxCount` / `sh:qualifiedValueShapesDisjoint` | `sh:QualifiedValueShapeConstraintComponent` | §4.7.3 | `QualifiedValueShape` | [`QualifiedValueShapeConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/shape_constraints/struct.QualifiedValueShapeConstraint.html) |
| `sh:closed` / `sh:ignoredProperties` | `sh:ClosedConstraintComponent`         | §4.8.1 | `Closed`             | [`ClosedConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/shape_constraints/struct.ClosedConstraint.html) |
| `sh:hasValue`               | `sh:HasValueConstraintComponent`                | §4.8.2 | `HasValue`           | [`HasValueConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/comparison_constraints/struct.HasValueConstraint.html) |
| `sh:in`                     | `sh:InConstraintComponent`                      | §4.8.3 | `In`                 | [`InConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/comparison_constraints/struct.InConstraint.html) |

[`Shape::property_shapes`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/struct.Shape.html#structfield.property_shapes

---

## 4. Node kinds (`sh:nodeKind`)

The values valid as `sh:nodeKind` (Core §4.1.3, Table 5) are mapped to the
[`NodeKind`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/value_constraints/enum.NodeKind.html)
enum:

| SHACL value                | Variant                         |
|----------------------------|---------------------------------|
| `sh:IRI`                   | `NodeKind::Iri`                 |
| `sh:BlankNode`             | `NodeKind::BlankNode`           |
| `sh:Literal`               | `NodeKind::Literal`             |
| `sh:BlankNodeOrIRI`        | `NodeKind::BlankNodeOrIri`      |
| `sh:BlankNodeOrLiteral`    | `NodeKind::BlankNodeOrLiteral`  |
| `sh:IRIOrLiteral`          | `NodeKind::IriOrLiteral`        |

---

## 5. SHACL-SPARQL (Part B of the W3C Recommendation)

SHACL-SPARQL is an extension of SHACL Core defined in part B of the
Recommendation. Sections 5 and 6 below refer to the part-B numbering.

| Construct                           | Spec section | Symbol |
|-------------------------------------|--------------|--------|
| `sh:sparql` / `sh:SPARQLConstraint` | §5           | [`SparqlConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql/struct.SparqlConstraint.html) (`Constraint::Sparql` variant) |
| `sh:select`                         | §5.1         | `SparqlQuery::Select` arm of `SparqlConstraint` |
| `sh:ask`                            | §5.1         | `SparqlQuery::Ask` arm of `SparqlConstraint` |
| `sh:prefixes` / `sh:declare`        | §5.2.2       | `SparqlConstraint::prefixes` / `oxirs_shacl::sparql_af::PrefixMap` |
| `sh:SPARQLConstraintComponent`      | §6           | [`AdvancedSparqlConstraint`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/sparql_constraint/struct.AdvancedSparqlConstraint.html) |
| Custom SPARQL constraint registry   | §6           | [`SparqlConstraintLibrary`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql/struct.SparqlConstraintLibrary.html) |

---

## 6. SHACL-AF (Advanced Features)

Section numbers refer to the [SHACL Advanced Features](https://www.w3.org/TR/shacl-af/)
W3C Working Group Note. `sh:SPARQLAskValidator` is defined in SHACL-SPARQL
part B §6.2, not in SHACL-AF.

| Construct                          | SHACL-AF section | Symbol |
|------------------------------------|------------------|--------|
| `sh:SPARQLTarget`                  | §3.1             | [`SparqlAfTarget`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql_af/sparql_target/struct.SparqlAfTarget.html) (`Target::Sparql` variant) |
| `sh:SPARQLTargetType`              | §3.2             | [`SparqlTargetType`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql_af/target_type/struct.SparqlTargetType.html) + [`SparqlTargetTypeRegistry`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql_af/target_type/struct.SparqlTargetTypeRegistry.html) |
| `sh:parameter`                     | SHACL-AF §5.2 / SHACL-SPARQL §6.1 | [`SparqlTargetParameter`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql_af/target_type/struct.SparqlTargetParameter.html), [`oxirs_shacl::constraint_parameter`] |
| `sh:SPARQLAskValidator`            | SHACL-SPARQL §6.2 | [`SparqlAskValidator`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql_af/ask_validator/struct.SparqlAskValidator.html) + [`SparqlAskValidatorBuilder`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql_af/ask_validator/struct.SparqlAskValidatorBuilder.html) |
| Target combinators (union, intersection, difference) | (extension) | [`Target::Union`], [`Target::Intersection`], [`Target::Difference`] |
| Conditional / hierarchical targets | (extension)      | [`Target::Conditional`], [`Target::Hierarchical`], [`Target::PathBased`] |
| Node expressions (`sh:if`, etc.)   | §6               | [`oxirs_shacl::node_expressions`], [`oxirs_shacl::node_expression_evaluator`] |
| Rules (`sh:rule`)                  | §8               | [`RuleEngine`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/advanced_features/struct.RuleEngine.html) |
| Functions (`sh:function`)          | §5               | [`ShaclFunction`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/advanced_features/struct.ShaclFunction.html), [`FunctionRegistry`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/advanced_features/struct.FunctionRegistry.html) |
| Expressions (`sh:expression`)      | §7               | [`ExpressionConstraintComponent`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/expression_constraint/struct.ExpressionConstraintComponent.html), [`ShaclExpression`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/expression_constraint/enum.ShaclExpression.html) |

[`oxirs_shacl::constraint_parameter`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraint_parameter/index.html
[`Target::Union`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.Union
[`Target::Intersection`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.Intersection
[`Target::Difference`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.Difference
[`Target::Conditional`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.Conditional
[`Target::Hierarchical`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.Hierarchical
[`Target::PathBased`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/enum.Target.html#variant.PathBased
[`oxirs_shacl::node_expressions`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/node_expressions/index.html
[`oxirs_shacl::node_expression_evaluator`]: https://docs.rs/oxirs-shacl/latest/oxirs_shacl/node_expression_evaluator/index.html

---

## 7. Validation pipeline

| Concept                            | Symbol                                                                                       |
|------------------------------------|-----------------------------------------------------------------------------------------------|
| Top-level validator                | [`ValidationEngine`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/validation/struct.ValidationEngine.html) |
| Validation configuration           | [`ValidationConfig`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/struct.ValidationConfig.html) |
| Validation strategy                | [`ValidationStrategy`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/optimization/integration/enum.ValidationStrategy.html) |
| Per-constraint context             | [`ConstraintContext`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/constraint_context/struct.ConstraintContext.html) |
| Constraint result                  | [`ConstraintEvaluationResult`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraints/constraint_context/enum.ConstraintEvaluationResult.html) |
| Violation entry                    | [`ValidationViolation`]                                                                       |
| Validation report                  | [`ValidationReport`]                                                                          |
| Report writers (Turtle/JSON-LD/etc.)| [`ReportFormat`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/report/enum.ReportFormat.html), [`ReportGenerator`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/report/struct.ReportGenerator.html) |
| Cached validation                  | [`ValidationCache`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/cache/struct.ValidationCache.html) |
| Parallel validation                | [`ParallelConstraintValidator`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/cache/struct.ParallelConstraintValidator.html) |
| Incremental validation             | [`IncrementalValidator`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/incremental/struct.IncrementalValidator.html) |
| Federated validation               | [`oxirs_shacl::federated_validation`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/federated_validation/index.html) |
| Optimisation engine                | [`ValidationOptimizationEngine`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/optimization/struct.ValidationOptimizationEngine.html), [`NegationOptimizer`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/optimization/struct.NegationOptimizer.html) |
| Severity handling                  | [`oxirs_shacl::severity_handler`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/severity_handler/index.html) |
| Message templates                  | [`oxirs_shacl::message_formatter`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/message_formatter/index.html) |
| Shape graph loader                 | [`oxirs_shacl::shape_graph_loader`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/shape_graph_loader/index.html) |
| Shape inheritance (`sh:and` chain) | [`oxirs_shacl::shape_inheritance`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/shape_inheritance/index.html), [`oxirs_shacl::constraint_inheritance`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/constraint_inheritance/index.html) |
| Entailment regimes                 | [`oxirs_shacl::entailment_regime`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/entailment_regime/index.html) |
| Datatype validation primitives     | [`oxirs_shacl::datatype_checker`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/datatype_checker/index.html) |
| Focus-node selection               | [`oxirs_shacl::focus_node_selector`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/focus_node_selector/index.html), [`TargetSelector`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/selector/struct.TargetSelector.html) |
| SHACL Compact Syntax parser        | [`oxirs_shacl::shaclc_parser`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/shaclc_parser/index.html) |
| Shape import (`owl:imports`)       | [`oxirs_shacl::shape_import`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/shape_import/index.html) |
| Shape versioning                   | [`oxirs_shacl::shape_versioning`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/shape_versioning/index.html) |

---

## 8. Identifiers and primitive types

| Concept                  | Symbol                                                        |
|--------------------------|---------------------------------------------------------------|
| Shape ID (IRI / blank)   | [`ShapeId`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/struct.ShapeId.html) |
| Constraint component ID  | [`ConstraintComponentId`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/struct.ConstraintComponentId.html) |
| Error type               | [`ShaclError`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/enum.ShaclError.html) |
| Result alias             | [`Result`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/type.Result.html) |
| Shape metadata           | [`ShapeMetadata`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/struct.ShapeMetadata.html) |

---

## 9. SHACL test suite integration

| Concept                                | Symbol                                                      |
|----------------------------------------|-------------------------------------------------------------|
| W3C SHACL test suite runner            | [`oxirs_shacl::w3c_test_suite`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/w3c_test_suite/index.html) |
| Enhanced suite (full conformance)      | [`oxirs_shacl::w3c_test_suite_enhanced`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/w3c_test_suite_enhanced/index.html) |
| Standalone shape testing harness       | [`oxirs_shacl::testing`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/testing/index.html) |

---

*Last updated: 2026-04-30. Spec sections refer to the canonical W3C
Recommendation/Note dated 2017-07-20 (Core) and 2017-06-08 (Advanced Features).*
