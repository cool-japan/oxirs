use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_core::{
    model::{Literal, NamedNode, Term},
    ConcreteStore, Store,
};
use oxirs_shacl::{
    constraints::*, shapes::ShapeFactory, Constraint, ConstraintComponentId, PropertyPath,
    Severity, Shape, ShapeId, ShapeType, Target, ValidationConfig, Validator,
};

fn create_test_data(size: usize) -> (ConcreteStore, Vec<Shape>) {
    let store = ConcreteStore::new().unwrap();
    let mut shapes = Vec::new();

    // Create test data in the store
    for i in 0..size {
        let person_iri = NamedNode::new(&format!("http://example.org/person{}", i)).unwrap();
        let name_literal = Term::Literal(Literal::new(&format!("Person {}", i)));
        let age_literal = Term::Literal(Literal::new(&format!("{}", 20 + (i % 60))));

        // Add person type
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let person_class = NamedNode::new("http://example.org/Person").unwrap();

        // Add name
        let name_predicate = NamedNode::new("http://example.org/name").unwrap();

        // Add age
        let age_predicate = NamedNode::new("http://example.org/age").unwrap();

        // TODO: Add triples to store when store API is available
    }

    // Create test shapes
    let person_class = NamedNode::new("http://example.org/Person").unwrap();
    let person_shape = ShapeFactory::node_shape_with_class(
        ShapeId::new("http://example.org/PersonShape"),
        person_class,
    );
    shapes.push(person_shape);

    // Create name property shape
    let name_path = PropertyPath::predicate(NamedNode::new("http://example.org/name").unwrap());
    let name_shape = ShapeFactory::string_property_shape(
        ShapeId::new("http://example.org/NameShape"),
        name_path,
        Some(1),                          // min length
        Some(100),                        // max length
        Some("^[A-Za-z ]+$".to_string()), // pattern
    );
    shapes.push(name_shape);

    // Create age property shape
    let age_path = PropertyPath::predicate(NamedNode::new("http://example.org/age").unwrap());
    let age_shape = ShapeFactory::cardinality_property_shape(
        ShapeId::new("http://example.org/AgeShape"),
        age_path,
        Some(1), // min count
        Some(1), // max count
    );
    shapes.push(age_shape);

    (store, shapes)
}

fn bench_validation_small(c: &mut Criterion) {
    let (store, shapes) = create_test_data(10);
    let mut validator = Validator::new();

    for shape in shapes {
        validator.add_shape(shape).unwrap();
    }

    c.bench_function("validation_small_10_items", |b| {
        b.iter(|| {
            let result = validator.validate_store(black_box(&store), None);
            black_box(result)
        })
    });
}

fn bench_validation_medium(c: &mut Criterion) {
    let (store, shapes) = create_test_data(100);
    let mut validator = Validator::new();

    for shape in shapes {
        validator.add_shape(shape).unwrap();
    }

    c.bench_function("validation_medium_100_items", |b| {
        b.iter(|| {
            let result = validator.validate_store(black_box(&store), None);
            black_box(result)
        })
    });
}

fn bench_validation_large(c: &mut Criterion) {
    let (store, shapes) = create_test_data(1000);
    let mut validator = Validator::new();

    for shape in shapes {
        validator.add_shape(shape).unwrap();
    }

    c.bench_function("validation_large_1000_items", |b| {
        b.iter(|| {
            let result = validator.validate_store(black_box(&store), None);
            black_box(result)
        })
    });
}

fn bench_constraint_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_validation");

    // Test different constraint types
    let constraints = vec![
        (
            "class",
            Constraint::Class(ClassConstraint {
                class_iri: NamedNode::new("http://example.org/Person").unwrap(),
            }),
        ),
        (
            "min_count",
            Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        ),
        (
            "max_count",
            Constraint::MaxCount(MaxCountConstraint { max_count: 10 }),
        ),
        (
            "min_length",
            Constraint::MinLength(MinLengthConstraint { min_length: 1 }),
        ),
        (
            "max_length",
            Constraint::MaxLength(MaxLengthConstraint { max_length: 100 }),
        ),
        (
            "pattern",
            Constraint::Pattern(PatternConstraint {
                pattern: "^[A-Za-z ]+$".to_string(),
                flags: None,
                message: None,
            }),
        ),
        (
            "node_kind",
            Constraint::NodeKind(NodeKindConstraint {
                node_kind: NodeKind::Literal,
            }),
        ),
    ];

    for (name, constraint) in constraints {
        group.bench_with_input(
            BenchmarkId::new("constraint", name),
            &constraint,
            |b, constraint| {
                b.iter(|| {
                    let result = constraint.validate();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn bench_shape_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_parsing");

    // Create different complexity RDF data for parsing
    let simple_shapes_ttl = r#"
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        
        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:datatype xsd:string ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
            ] .
    "#;

    let complex_shapes_ttl = r#"
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        
        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:datatype xsd:string ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:minLength 1 ;
                sh:maxLength 100 ;
                sh:pattern "^[A-Za-z ]+$" ;
            ] ;
            sh:property [
                sh:path ex:age ;
                sh:datatype xsd:integer ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:minInclusive 0 ;
                sh:maxInclusive 150 ;
            ] ;
            sh:property [
                sh:path ex:email ;
                sh:datatype xsd:string ;
                sh:minCount 0 ;
                sh:maxCount 1 ;
                sh:pattern "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$" ;
            ] .
            
        ex:CompanyShape a sh:NodeShape ;
            sh:targetClass ex:Company ;
            sh:property [
                sh:path ex:name ;
                sh:datatype xsd:string ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
            ] ;
            sh:property [
                sh:path ex:employees ;
                sh:class ex:Person ;
                sh:minCount 1 ;
            ] .
    "#;

    group.bench_function("simple_shapes", |b| {
        b.iter(|| {
            let mut validator = Validator::new();
            let result = validator.load_shapes_from_rdf(
                black_box(simple_shapes_ttl),
                "turtle",
                Some("http://example.org/"),
            );
            black_box(result)
        });
    });

    group.bench_function("complex_shapes", |b| {
        b.iter(|| {
            let mut validator = Validator::new();
            let result = validator.load_shapes_from_rdf(
                black_box(complex_shapes_ttl),
                "turtle",
                Some("http://example.org/"),
            );
            black_box(result)
        });
    });

    group.finish();
}

fn bench_property_path_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_path_evaluation");

    let store = ConcreteStore::new().unwrap();
    // TODO: Add test data to store when API is available

    let paths = vec![
        (
            "simple",
            PropertyPath::predicate(NamedNode::new("http://example.org/name").unwrap()),
        ),
        (
            "inverse",
            PropertyPath::inverse(PropertyPath::predicate(
                NamedNode::new("http://example.org/knows").unwrap(),
            )),
        ),
        (
            "sequence",
            PropertyPath::sequence(vec![
                PropertyPath::predicate(NamedNode::new("http://example.org/worksFor").unwrap()),
                PropertyPath::predicate(NamedNode::new("http://example.org/name").unwrap()),
            ]),
        ),
        (
            "alternative",
            PropertyPath::alternative(vec![
                PropertyPath::predicate(NamedNode::new("http://example.org/name").unwrap()),
                PropertyPath::predicate(NamedNode::new("http://example.org/label").unwrap()),
            ]),
        ),
    ];

    for (name, path) in paths {
        group.bench_with_input(BenchmarkId::new("path", name), &path, |b, path| {
            b.iter(|| {
                // TODO: Implement path evaluation benchmark when API is available
                black_box(path)
            });
        });
    }

    group.finish();
}

fn bench_validation_config_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_config_impact");

    let (store, shapes) = create_test_data(100);
    let mut validator = Validator::new();

    for shape in shapes {
        validator.add_shape(shape).unwrap();
    }

    let configs = vec![
        ("default", ValidationConfig::default()),
        (
            "fail_fast",
            ValidationConfig {
                fail_fast: true,
                ..ValidationConfig::default()
            },
        ),
        (
            "max_violations_10",
            ValidationConfig {
                max_violations: 10,
                ..ValidationConfig::default()
            },
        ),
        (
            "exclude_warnings",
            ValidationConfig {
                include_warnings: false,
                ..ValidationConfig::default()
            },
        ),
        (
            "parallel",
            ValidationConfig {
                parallel: true,
                ..ValidationConfig::default()
            },
        ),
    ];

    for (name, config) in configs {
        group.bench_with_input(BenchmarkId::new("config", name), &config, |b, config| {
            b.iter(|| {
                let result = validator.validate_store(black_box(&store), Some(config.clone()));
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_validation_small,
    bench_validation_medium,
    bench_validation_large,
    bench_constraint_validation,
    bench_shape_parsing,
    bench_property_path_evaluation,
    bench_validation_config_impact
);
criterion_main!(benches);
