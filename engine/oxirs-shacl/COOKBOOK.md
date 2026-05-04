# OxiRS SHACL Cookbook

A task-oriented reference for the most common shape patterns. Every recipe
shows three things:

1. The Turtle representation that you would write in a SHACL shapes graph.
2. The equivalent Rust API call to construct (or just describe) the constraint
   programmatically through `oxirs_shacl`.
3. A short note on the validation semantics so the example is unambiguous.

The companion document `SPEC_MAPPING.md` is the exhaustive table of every
SHACL Core / SHACL-AF construct mapped to the Rust symbol that implements it.

> **Conventions.** All Turtle examples use the prefixes
>
> ```turtle
> @prefix sh:   <http://www.w3.org/ns/shacl#> .
> @prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
> @prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
> @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
> @prefix ex:   <http://example.org/> .
> ```

---

## 1. Cardinality

### 1.1 Make a property mandatory (`sh:minCount`)

```turtle
ex:PersonShape
    a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:hasName ;
        sh:minCount 1 ;
    ] .
```

```rust
use oxirs_shacl::constraints::cardinality_constraints::MinCountConstraint;

let mandatory = MinCountConstraint { min_count: 1 };
```

A value count `< 1` produces a violation with
`sh:sourceConstraintComponent = sh:MinCountConstraintComponent`.

### 1.2 Cap a multi-valued property (`sh:maxCount`)

```turtle
ex:PersonShape
    sh:property [
        sh:path ex:emergencyContact ;
        sh:maxCount 3 ;
    ] .
```

```rust
use oxirs_shacl::constraints::cardinality_constraints::MaxCountConstraint;

let cap = MaxCountConstraint { max_count: 3 };
```

A value count `> 3` is reported. Setting `max_count: 0` makes the property
forbidden.

### 1.3 Functional properties (`sh:minCount 1` + `sh:maxCount 1`)

```turtle
ex:PersonShape
    sh:property [
        sh:path ex:hasSocialSecurityNumber ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .
```

```rust
use oxirs_shacl::constraints::cardinality_constraints::{MaxCountConstraint, MinCountConstraint};

let exactly_one_min = MinCountConstraint { min_count: 1 };
let exactly_one_max = MaxCountConstraint { max_count: 1 };
```

---

## 2. String constraints

### 2.1 Length bounds (`sh:minLength`, `sh:maxLength`)

```turtle
ex:PersonShape
    sh:property [
        sh:path ex:nickname ;
        sh:minLength 3 ;
        sh:maxLength 32 ;
    ] .
```

```rust
use oxirs_shacl::constraints::string_constraints::{MaxLengthConstraint, MinLengthConstraint};

let min_len = MinLengthConstraint { min_length: 3 };
let max_len = MaxLengthConstraint { max_length: 32 };
```

Length is measured in Unicode scalar values, not bytes. Non-literal value
nodes always fail these constraints.

### 2.2 Regex pattern (`sh:pattern`, `sh:flags`)

```turtle
ex:PersonShape
    sh:property [
        sh:path ex:email ;
        sh:pattern "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$" ;
        sh:flags   "i" ;
        sh:message "Must be an email address." ;
    ] .
```

```rust
use oxirs_shacl::constraints::string_constraints::PatternConstraint;

let email = PatternConstraint {
    pattern: r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$".to_string(),
    flags:   Some("i".to_string()),
    message: Some("Must be an email address.".to_string()),
};
```

`flags` accepts the standard regex flags (`i`, `m`, `s`, `x`); the regex is
compiled by the [`regex`](https://docs.rs/regex) crate, so its syntax applies.

### 2.3 Allowed languages (`sh:languageIn`)

```turtle
ex:LabelShape
    sh:property [
        sh:path rdfs:label ;
        sh:languageIn ( "en" "fr" "de" ) ;
    ] .
```

```rust
use oxirs_shacl::constraints::string_constraints::LanguageInConstraint;

let langs = LanguageInConstraint {
    languages: vec!["en".into(), "fr".into(), "de".into()],
};
```

A literal whose language tag is missing or not in the list is reported.
BCP-47 prefix matching is applied (`en` matches `en-US`).

### 2.4 Unique language tags (`sh:uniqueLang`)

```turtle
ex:LabelShape
    sh:property [
        sh:path rdfs:label ;
        sh:uniqueLang true ;
    ] .
```

```rust
use oxirs_shacl::constraints::string_constraints::UniqueLangConstraint;

let unique = UniqueLangConstraint { unique_lang: true };
```

Reports a violation if two values share the same language tag.

---

## 3. Datatype, class, and node-kind constraints

### 3.1 Datatype (`sh:datatype`)

```turtle
ex:PersonShape
    sh:property [
        sh:path ex:age ;
        sh:datatype xsd:integer ;
    ] .
```

```rust
use oxirs_core::model::NamedNode;
use oxirs_shacl::constraints::value_constraints::DatatypeConstraint;

let int_only = DatatypeConstraint {
    datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
};
# Ok::<_, oxirs_core::OxirsError>(())
```

The constraint passes only for plain literals carrying that datatype IRI.

### 3.2 Class (`sh:class`)

```turtle
ex:PersonShape
    sh:property [
        sh:path ex:bestFriend ;
        sh:class ex:Person ;
    ] .
```

```rust
use oxirs_core::model::NamedNode;
use oxirs_shacl::constraints::value_constraints::ClassConstraint;

let person_only = ClassConstraint {
    class_iri: NamedNode::new("http://example.org/Person")?,
};
# Ok::<_, oxirs_core::OxirsError>(())
```

### 3.3 Node kind (`sh:nodeKind`)

```turtle
ex:PersonShape
    sh:property [
        sh:path ex:homepage ;
        sh:nodeKind sh:IRI ;
    ] .
```

```rust
use oxirs_shacl::constraints::value_constraints::{NodeKind, NodeKindConstraint};

let iri_only = NodeKindConstraint { node_kind: NodeKind::Iri };
```

Pick the variant that mirrors the SHACL keyword:
`Iri`, `BlankNode`, `Literal`, `BlankNodeOrIri`, `BlankNodeOrLiteral`,
`IriOrLiteral`.

---

## 4. Value-list and comparison constraints

### 4.1 Closed enumeration (`sh:in`)

```turtle
ex:OrderShape
    sh:property [
        sh:path ex:status ;
        sh:in ( "pending" "shipped" "delivered" ) ;
    ] .
```

```rust
use oxirs_core::model::{Literal, Term};
use oxirs_shacl::constraints::comparison_constraints::InConstraint;

let statuses = InConstraint {
    values: vec![
        Term::Literal(Literal::new_simple_literal("pending")),
        Term::Literal(Literal::new_simple_literal("shipped")),
        Term::Literal(Literal::new_simple_literal("delivered")),
    ],
};
```

### 4.2 Required value (`sh:hasValue`)

```turtle
ex:LegalShape
    sh:property [
        sh:path ex:agreedTo ;
        sh:hasValue ex:TermsOfService ;
    ] .
```

```rust
use oxirs_core::model::{NamedNode, Term};
use oxirs_shacl::constraints::comparison_constraints::HasValueConstraint;

let agreement = HasValueConstraint {
    value: Term::NamedNode(NamedNode::new("http://example.org/TermsOfService")?),
};
# Ok::<_, oxirs_core::OxirsError>(())
```

### 4.3 Numeric range (`sh:minInclusive`, `sh:maxExclusive`)

```turtle
ex:DiscountShape
    sh:property [
        sh:path ex:percent ;
        sh:datatype xsd:decimal ;
        sh:minInclusive 0 ;
        sh:maxExclusive 100 ;
    ] .
```

```rust
use oxirs_core::model::Literal;
use oxirs_shacl::constraints::range_constraints::{MaxExclusiveConstraint, MinInclusiveConstraint};

let lower = MinInclusiveConstraint {
    min_value: Literal::new_simple_literal("0"),
};
let upper = MaxExclusiveConstraint {
    max_value: Literal::new_simple_literal("100"),
};
```

### 4.4 Property comparisons (`sh:lessThan`)

```turtle
ex:PeriodShape
    sh:property [
        sh:path ex:startDate ;
        sh:lessThan ex:endDate ;
    ] .
```

```rust
use oxirs_core::model::{NamedNode, Term};
use oxirs_shacl::constraints::comparison_constraints::LessThanConstraint;

let chronological = LessThanConstraint {
    property: Term::NamedNode(NamedNode::new("http://example.org/endDate")?),
};
# Ok::<_, oxirs_core::OxirsError>(())
```

---

## 5. Qualified value shapes

`sh:qualifiedValueShape` lets you say "at least N values must conform to a
specific shape" without forcing all values to. It's the canonical way to model
"a Person must have at least one phone number that is a mobile number".

```turtle
ex:PersonShape
    a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:phone ;
        sh:qualifiedValueShape ex:MobilePhoneShape ;
        sh:qualifiedMinCount 1 ;
    ] .

ex:MobilePhoneShape
    a sh:NodeShape ;
    sh:property [
        sh:path ex:kind ;
        sh:hasValue "mobile" ;
    ] .
```

```rust
use oxirs_shacl::{ShapeId, constraints::shape_constraints::QualifiedValueShapeConstraint};

let qvs = QualifiedValueShapeConstraint::new(ShapeId::new("http://example.org/MobilePhoneShape"))
    .with_qualified_min_count(1);
```

For the disjoint flavour (so that a value already counted by a sibling
qualified shape is excluded):

```rust
use oxirs_shacl::{ShapeId, constraints::shape_constraints::QualifiedValueShapeConstraint};

let qvs = QualifiedValueShapeConstraint::new(ShapeId::new("http://example.org/AdultShape"))
    .with_qualified_min_count(1)
    .with_qualified_max_count(2)
    .with_qualified_value_shapes_disjoint(true);
```

`validate()` will reject a constraint that has neither min nor max set, or
where `min > max`.

---

## 6. Logical combinators

### 6.1 Conjunction (`sh:and`)

```turtle
ex:AdultPersonShape
    a sh:NodeShape ;
    sh:and ( ex:PersonShape ex:AdultShape ) .
```

```rust
use oxirs_shacl::{ShapeId, constraints::logical_constraints::AndConstraint};

let both = AndConstraint::new(vec![
    ShapeId::new("http://example.org/PersonShape"),
    ShapeId::new("http://example.org/AdultShape"),
]);
```

### 6.2 Disjunction (`sh:or`)

```turtle
ex:ContactShape
    a sh:NodeShape ;
    sh:or ( ex:PersonShape ex:OrganizationShape ) .
```

```rust
use oxirs_shacl::{ShapeId, constraints::logical_constraints::OrConstraint};

let either = OrConstraint::new(vec![
    ShapeId::new("http://example.org/PersonShape"),
    ShapeId::new("http://example.org/OrganizationShape"),
]);
```

### 6.3 Exclusive or (`sh:xone`)

```turtle
ex:RoleShape
    a sh:NodeShape ;
    sh:xone ( ex:CustomerShape ex:EmployeeShape ) .
```

```rust
use oxirs_shacl::{ShapeId, constraints::logical_constraints::XoneConstraint};

let exactly_one = XoneConstraint::new(vec![
    ShapeId::new("http://example.org/CustomerShape"),
    ShapeId::new("http://example.org/EmployeeShape"),
]);
```

### 6.4 Negation (`sh:not`)

```turtle
ex:NotMinorShape
    a sh:NodeShape ;
    sh:not ex:MinorShape .
```

```rust
use oxirs_shacl::{ShapeId, constraints::logical_constraints::NotConstraint};

let not_minor = NotConstraint::new(ShapeId::new("http://example.org/MinorShape"));
```

`sh:not` is the most expensive combinator because it requires negation over
shape conformance; the engine routes it through
`oxirs_shacl::optimization::NegationOptimizer`.

---

## 7. Target chains

A *target chain* is what you get when you combine multiple SHACL target
declarations on a single shape. Targets are unioned: a focus node is in scope
when it matches any one of them.

### 7.1 Class target (`sh:targetClass`)

```turtle
ex:PersonShape
    a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:targetClass foaf:Person .
```

```rust
use oxirs_core::model::NamedNode;
use oxirs_shacl::{Shape, ShapeId, targets::Target};

let mut shape = Shape::node_shape(ShapeId::new("http://example.org/PersonShape"));
shape.add_target(Target::Class(NamedNode::new("http://example.org/Person")?));
shape.add_target(Target::Class(NamedNode::new("http://xmlns.com/foaf/0.1/Person")?));
# Ok::<_, oxirs_core::OxirsError>(())
```

### 7.2 Subjects-of and objects-of targets

```turtle
ex:HasNameShape
    a sh:NodeShape ;
    sh:targetSubjectsOf ex:hasName .

ex:NameValueShape
    a sh:NodeShape ;
    sh:targetObjectsOf ex:hasName .
```

```rust
use oxirs_core::model::NamedNode;
use oxirs_shacl::targets::Target;

let subjects = Target::SubjectsOf(NamedNode::new("http://example.org/hasName")?);
let objects  = Target::ObjectsOf(NamedNode::new("http://example.org/hasName")?);
# Ok::<_, oxirs_core::OxirsError>(())
```

### 7.3 Implicit class target (shape IRI used as `rdf:type`)

A `sh:NodeShape` whose IRI is also used as an `rdf:type` for instances acts as
its own implicit class target.

```rust
use oxirs_core::model::NamedNode;
use oxirs_shacl::targets::Target;

let implicit = Target::Implicit(NamedNode::new("http://example.org/PersonShape")?);
# Ok::<_, oxirs_core::OxirsError>(())
```

### 7.4 Target intersection / difference

The engine supports the SHACL-AF target combinators (
[`Target::Union`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/targets/types/struct.TargetUnion.html),
`Intersection`, `Difference`):

```rust
use oxirs_core::model::NamedNode;
use oxirs_shacl::targets::{Target, types::{TargetIntersection, TargetDifference}};

let active_persons = Target::Intersection(TargetIntersection {
    targets: vec![
        Target::Class(NamedNode::new("http://example.org/Person")?),
        Target::SubjectsOf(NamedNode::new("http://example.org/lastLoggedIn")?),
    ],
    optimization_hints: None,
});

let non_admin_persons = Target::Difference(TargetDifference {
    primary_target:   Box::new(Target::Class(NamedNode::new("http://example.org/Person")?)),
    exclusion_target: Box::new(Target::Class(NamedNode::new("http://example.org/Admin")?)),
});
# Ok::<_, oxirs_core::OxirsError>(())
```

---

## 8. SHACL-AF SPARQL constraints

For constraints that cannot be expressed in pure SHACL Core, SHACL-AF allows
arbitrary SPARQL queries.

### 8.1 SELECT-style SPARQL constraint

```turtle
ex:UniqueShape
    a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:sparql [
        sh:message "Two people share the same SSN: {?other}" ;
        sh:prefixes ex: ;
        sh:select """
            SELECT $this ?other WHERE {
                $this ex:ssn ?ssn .
                ?other ex:ssn ?ssn .
                FILTER ( $this != ?other )
            }
        """ ;
    ] .
```

```rust
use oxirs_shacl::sparql::{SparqlConstraint, SparqlQuery};

let dup_ssn = SparqlConstraint::new(SparqlQuery::Select(
    r#"
    SELECT $this ?other WHERE {
        $this ex:ssn ?ssn .
        ?other ex:ssn ?ssn .
        FILTER ( $this != ?other )
    }
    "#.to_string(),
))
.with_message("Two people share the same SSN: {?other}");
```

Each row returned by the SELECT is a violation, with `$this` providing the
focus node and the rest of the variables filling the message template.

### 8.2 ASK-style validator (`sh:SPARQLAskValidator`)

```rust
use oxirs_shacl::sparql_af::ask_validator::SparqlAskValidatorBuilder;

let only_adults = SparqlAskValidatorBuilder::new()
    .ask_query("ASK { $this ex:age ?a . FILTER (?a >= 18) }")
    .message("Subject must be an adult")
    .build()?;
# Ok::<_, oxirs_shacl::sparql_af::SparqlAfError>(())
```

A `false` result from the ASK query triggers a violation. The validator
substitutes `$this` (and any other parameter binding from the
[`SubstitutionContext`](https://docs.rs/oxirs-shacl/latest/oxirs_shacl/sparql_af/struct.SubstitutionContext.html))
before execution.

### 8.3 SPARQL target (`sh:SPARQLTarget`)

```turtle
ex:RecentlyActiveShape
    a sh:NodeShape ;
    sh:target [
        a sh:SPARQLTarget ;
        sh:select """
            SELECT $this WHERE {
                $this ex:lastLoginDate ?d .
                FILTER (?d >= "2024-01-01"^^xsd:date)
            }
        """ ;
    ] .
```

```rust
use oxirs_shacl::sparql_af::sparql_target::SparqlAfTarget;

let recently_active = SparqlAfTarget::new(
    r#"SELECT $this WHERE {
        $this ex:lastLoginDate ?d .
        FILTER (?d >= "2024-01-01"^^xsd:date)
    }"#,
);
```

The query must bind `$this` for every row; the engine treats each binding as
one focus node.

### 8.4 SPARQL target type (`sh:SPARQLTargetType`)

A `sh:SPARQLTargetType` is a parameterised target template. Define it once,
instantiate it per-shape:

```rust
use oxirs_shacl::sparql_af::target_type::{
    SparqlTargetParameter, SparqlTargetType, SparqlTargetTypeRegistry,
};

let mut registry = SparqlTargetTypeRegistry::new();

let by_class = SparqlTargetType::new(
    "http://example.org/InstanceOf",
    "SELECT ?this WHERE { ?this rdf:type ?className }",
)
.with_parameter(SparqlTargetParameter::new("className"));

registry.register(by_class);
```

Each `SparqlTargetTypeInstance` then binds the parameters and yields a
concrete target on demand.

---

## 9. Putting it together — a minimal validation run

```rust,no_run
use oxirs_core::Store;
use oxirs_shacl::{
    Shape, ShapeId, ValidationConfig, ValidationStrategy,
    constraints::{cardinality_constraints::MinCountConstraint, Constraint},
    targets::Target,
    ConstraintComponentId,
    validation::ValidationEngine,
};
use oxirs_core::model::NamedNode;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut shape = Shape::node_shape(ShapeId::new("http://example.org/PersonShape"));
shape.add_target(Target::Class(NamedNode::new("http://example.org/Person")?));
shape.add_constraint(
    ConstraintComponentId::new("sh:MinCountConstraintComponent"),
    Constraint::MinCount(MinCountConstraint { min_count: 1 }),
);

let mut shapes = indexmap::IndexMap::new();
shapes.insert(shape.id.clone(), shape);

let config = ValidationConfig::default()
    .with_strategy(ValidationStrategy::Optimized);

// Bring your own data store implementing oxirs_core::Store.
// let store: &dyn Store = ...;
// let mut engine = ValidationEngine::new(&shapes, config);
// let report = engine.validate(store)?;
# Ok(())
# }
```

Refer to `examples/basic_validation.rs` and `examples/advanced_shapes.rs` for
runnable end-to-end examples that load shapes from Turtle files.
