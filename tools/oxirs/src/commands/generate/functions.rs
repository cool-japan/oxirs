//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::CommandResult;
use super::{owl, rdfs, shacl, types::*};
use crate::cli::logging::{DataLogger, PerfLogger};
use crate::cli::{format_bytes, format_duration, format_number};
use crate::cli::{progress::helpers, CliContext};
use oxirs_core::format::{RdfFormat, RdfParser, RdfSerializer};
use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use oxirs_core::RdfTerm;
use scirs2_core::random::Random;
use scirs2_core::Rng;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
/// Generate a synthetic RDF dataset
pub async fn run(
    output: PathBuf,
    size: String,
    dataset_type: String,
    format: String,
    seed: Option<u64>,
    schema: Option<PathBuf>,
) -> CommandResult {
    let ctx = CliContext::new();
    if let Some(schema_file) = schema {
        return run_schema_based_generation(output, size, schema_file, format, seed).await;
    }
    ctx.info("Generating synthetic RDF dataset");
    ctx.info(&format!("Output file: {}", output.display()));
    let size_enum = DatasetSize::from_string(&size)?;
    let type_enum = DatasetType::from_string(&dataset_type)?;
    let triple_count = size_enum.triple_count();
    ctx.info(&format!("Size: {} ({} triples)", size, triple_count));
    ctx.info(&format!("Type: {}", dataset_type));
    ctx.info(&format!("Format: {}", format));
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let mut data_logger = DataLogger::new("generate", output.to_str().unwrap_or("unknown"));
    let mut perf_logger = PerfLogger::new(format!("generate_{}", dataset_type));
    perf_logger.add_metadata("size", size.clone());
    perf_logger.add_metadata("type", dataset_type.clone());
    perf_logger.add_metadata("format", &format);
    if let Some(s) = seed {
        perf_logger.add_metadata("seed", s.to_string());
    }
    let rdf_format = parse_rdf_format(&format)?;
    let progress = helpers::query_progress();
    progress.set_message("Generating RDF triples");
    let quads = match type_enum {
        DatasetType::Rdf => generate_random_rdf(&mut rng, triple_count),
        DatasetType::Graph => generate_graph_structure(&mut rng, triple_count),
        DatasetType::Semantic => generate_semantic_data(&mut rng, triple_count),
        DatasetType::Bibliographic => generate_bibliographic_data(&mut rng, triple_count),
        DatasetType::Geographic => generate_geographic_data(&mut rng, triple_count),
        DatasetType::Organizational => generate_organizational_data(&mut rng, triple_count),
    };
    progress.set_message("Writing to file");
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("ex", "http://example.org/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        match serializer.serialize_quad(quad.as_ref()) {
            Ok(_) => written += 1,
            Err(e) => {
                return Err(format!("Failed to serialize quad: {}", e).into());
            }
        }
    }
    serializer
        .finish()
        .map_err(|e| format!("Failed to finalize serialization: {}", e))?;
    progress.finish_with_message("Dataset generation complete");
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    data_logger.update_progress(file_size, written as u64);
    data_logger.complete();
    perf_logger.add_metadata("triple_count", written);
    perf_logger.complete(Some(5000));
    use crate::cli::{format_bytes, format_duration, format_number};
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "✓ Dataset generated in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Triples generated: {}",
        format_number(written as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));
    if duration.as_secs_f64() > 0.0 {
        let rate = written as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Generation rate: {} triples/second",
            format_number(rate as u64)
        ));
    }
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}
/// Generate random RDF triples
fn generate_random_rdf<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
    let mut quads = Vec::with_capacity(count);
    let predicates = [
        "name",
        "age",
        "email",
        "address",
        "phone",
        "description",
        "value",
        "status",
        "created",
        "modified",
    ];
    for _i in 0..count {
        let subject_id = rng.random_range(0..count.max(100));
        let predicate_idx = rng.random_range(0..predicates.len());
        let object_value = rng.random_range(0..10000);
        let subject = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/resource/{}", subject_id))
                .expect("Valid IRI"),
        );
        let predicate = NamedNode::new(format!("http://example.org/{}", predicates[predicate_idx]))
            .expect("Valid IRI");
        let object = if rng.random_bool(0.5) {
            Term::Literal(Literal::new_simple_literal(format!(
                "value_{}",
                object_value
            )))
        } else {
            let xsd_integer =
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("Valid IRI");
            Term::Literal(Literal::new_typed_literal(
                object_value.to_string(),
                xsd_integer,
            ))
        };
        quads.push(Quad::new(
            subject,
            predicate,
            object,
            GraphName::DefaultGraph,
        ));
    }
    quads
}
/// Generate graph structure (nodes and edges)
fn generate_graph_structure<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
    let mut quads = Vec::with_capacity(count);
    let node_count = (count as f64).sqrt() as usize;
    let edge_types = ["connected", "linked", "related", "parent", "child"];
    for _i in 0..count {
        let source_id = rng.random_range(0..node_count);
        let target_id = rng.random_range(0..node_count);
        let edge_type_idx = rng.random_range(0..edge_types.len());
        let subject = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/node/{}", source_id)).expect("Valid IRI"),
        );
        let predicate = NamedNode::new(format!("http://example.org/{}", edge_types[edge_type_idx]))
            .expect("Valid IRI");
        let object = Term::NamedNode(
            NamedNode::new(format!("http://example.org/node/{}", target_id)).expect("Valid IRI"),
        );
        quads.push(Quad::new(
            subject,
            predicate,
            object,
            GraphName::DefaultGraph,
        ));
    }
    quads
}
/// Generate semantic web data (classes, properties, instances)
fn generate_semantic_data<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
    let mut quads = Vec::with_capacity(count);
    let classes = ["Person", "Organization", "Place", "Event", "Thing"];
    let properties = ["name", "description", "location", "date", "type"];
    let class_count = (count as f64 * 0.3) as usize;
    for i in 0..class_count {
        let class_idx = rng.random_range(0..classes.len());
        let instance_id = i;
        let subject = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/instance/{}", instance_id))
                .expect("Valid IRI"),
        );
        let predicate =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("Valid IRI");
        let object = Term::NamedNode(
            NamedNode::new(format!("http://example.org/class/{}", classes[class_idx]))
                .expect("Valid IRI"),
        );
        quads.push(Quad::new(
            subject,
            predicate,
            object,
            GraphName::DefaultGraph,
        ));
    }
    let property_count = count - class_count;
    for _i in 0..property_count {
        let instance_id = rng.random_range(0..class_count.max(1));
        let property_idx = rng.random_range(0..properties.len());
        let subject = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/instance/{}", instance_id))
                .expect("Valid IRI"),
        );
        let predicate = NamedNode::new(format!("http://example.org/{}", properties[property_idx]))
            .expect("Valid IRI");
        let object = Term::Literal(Literal::new_simple_literal(format!(
            "value_{}",
            rng.random_range(0..1000)
        )));
        quads.push(Quad::new(
            subject,
            predicate,
            object,
            GraphName::DefaultGraph,
        ));
    }
    quads
}
/// Generate bibliographic data (books, authors, publishers, citations)
fn generate_bibliographic_data<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
    let mut quads = Vec::with_capacity(count);
    let first_names = [
        "John", "Jane", "Alice", "Bob", "Carol", "David", "Emma", "Frank",
    ];
    let last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    ];
    let publishers = [
        "Academic Press",
        "MIT Press",
        "Springer",
        "Elsevier",
        "Wiley",
        "Oxford",
    ];
    let subjects = [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Biology",
        "Chemistry",
        "History",
    ];
    let num_authors = (count as f64 * 0.2) as usize;
    let num_books = (count as f64 * 0.3) as usize;
    let num_citations = count - num_authors - num_books;
    for i in 0..num_authors {
        let author_uri = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/author/{}", i)).expect("Valid IRI"),
        );
        let first = first_names[rng.random_range(0..first_names.len())];
        let last = last_names[rng.random_range(0..last_names.len())];
        quads.push(Quad::new(
            author_uri.clone(),
            NamedNode::new("http://xmlns.com/foaf/0.1/firstName").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(first)),
            GraphName::DefaultGraph,
        ));
        quads.push(Quad::new(
            author_uri.clone(),
            NamedNode::new("http://xmlns.com/foaf/0.1/lastName").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(last)),
            GraphName::DefaultGraph,
        ));
        quads.push(Quad::new(
            author_uri,
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("Valid IRI"),
            Term::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/Person").expect("Valid IRI")),
            GraphName::DefaultGraph,
        ));
    }
    for i in 0..num_books {
        let book_uri = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/book/{}", i)).expect("Valid IRI"),
        );
        quads.push(Quad::new(
            book_uri.clone(),
            NamedNode::new("http://purl.org/dc/terms/title").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(format!("Book Title {}", i))),
            GraphName::DefaultGraph,
        ));
        let author_id = rng.random_range(0..num_authors.max(1));
        quads.push(Quad::new(
            book_uri.clone(),
            NamedNode::new("http://purl.org/dc/terms/creator").expect("Valid IRI"),
            Term::NamedNode(
                NamedNode::new(format!("http://example.org/author/{}", author_id))
                    .expect("Valid IRI"),
            ),
            GraphName::DefaultGraph,
        ));
        let pub_name = publishers[rng.random_range(0..publishers.len())];
        quads.push(Quad::new(
            book_uri.clone(),
            NamedNode::new("http://purl.org/dc/terms/publisher").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(pub_name)),
            GraphName::DefaultGraph,
        ));
        let subject = subjects[rng.random_range(0..subjects.len())];
        quads.push(Quad::new(
            book_uri.clone(),
            NamedNode::new("http://purl.org/dc/terms/subject").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(subject)),
            GraphName::DefaultGraph,
        ));
        let year = 1950 + rng.random_range(0..75);
        let xsd_integer =
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("Valid IRI");
        quads.push(Quad::new(
            book_uri,
            NamedNode::new("http://purl.org/dc/terms/date").expect("Valid IRI"),
            Term::Literal(Literal::new_typed_literal(year.to_string(), xsd_integer)),
            GraphName::DefaultGraph,
        ));
    }
    for _i in 0..num_citations {
        let citing_book = rng.random_range(0..num_books.max(1));
        let cited_book = rng.random_range(0..num_books.max(1));
        if citing_book != cited_book {
            quads.push(Quad::new(
                Subject::NamedNode(
                    NamedNode::new(format!("http://example.org/book/{}", citing_book))
                        .expect("Valid IRI"),
                ),
                NamedNode::new("http://purl.org/dc/terms/references").expect("Valid IRI"),
                Term::NamedNode(
                    NamedNode::new(format!("http://example.org/book/{}", cited_book))
                        .expect("Valid IRI"),
                ),
                GraphName::DefaultGraph,
            ));
        }
    }
    quads
}
/// Generate geographic data (places, coordinates, addresses)
fn generate_geographic_data<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
    let mut quads = Vec::with_capacity(count);
    let cities = [
        "New York", "London", "Tokyo", "Paris", "Sydney", "Berlin", "Mumbai", "Toronto",
    ];
    let countries = [
        "USA",
        "UK",
        "Japan",
        "France",
        "Australia",
        "Germany",
        "India",
        "Canada",
    ];
    let street_types = ["Street", "Avenue", "Road", "Boulevard", "Lane", "Drive"];
    let num_places = (count as f64 * 0.25) as usize;
    for i in 0..num_places {
        let place_uri = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/place/{}", i)).expect("Valid IRI"),
        );
        quads.push(Quad::new(
            place_uri.clone(),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("Valid IRI"),
            Term::NamedNode(NamedNode::new("http://schema.org/Place").expect("Valid IRI")),
            GraphName::DefaultGraph,
        ));
        let city = cities[rng.random_range(0..cities.len())];
        quads.push(Quad::new(
            place_uri.clone(),
            NamedNode::new("http://schema.org/name").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(city)),
            GraphName::DefaultGraph,
        ));
        let street_num = rng.random_range(1..9999);
        let street_type = street_types[rng.random_range(0..street_types.len())];
        quads.push(Quad::new(
            place_uri.clone(),
            NamedNode::new("http://schema.org/streetAddress").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(format!(
                "{} Main {}",
                street_num, street_type
            ))),
            GraphName::DefaultGraph,
        ));
        let country = countries[rng.random_range(0..countries.len())];
        quads.push(Quad::new(
            place_uri.clone(),
            NamedNode::new("http://schema.org/addressCountry").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(country)),
            GraphName::DefaultGraph,
        ));
        let lat = -90.0 + rng.random_range(0..180) as f64;
        let xsd_double =
            NamedNode::new("http://www.w3.org/2001/XMLSchema#double").expect("Valid IRI");
        quads.push(Quad::new(
            place_uri.clone(),
            NamedNode::new("http://www.w3.org/2003/01/geo/wgs84_pos#lat").expect("Valid IRI"),
            Term::Literal(Literal::new_typed_literal(
                format!("{:.6}", lat),
                xsd_double.clone(),
            )),
            GraphName::DefaultGraph,
        ));
        let lon = -180.0 + rng.random_range(0..360) as f64;
        quads.push(Quad::new(
            place_uri,
            NamedNode::new("http://www.w3.org/2003/01/geo/wgs84_pos#long").expect("Valid IRI"),
            Term::Literal(Literal::new_typed_literal(
                format!("{:.6}", lon),
                xsd_double,
            )),
            GraphName::DefaultGraph,
        ));
    }
    quads
}
/// Generate organizational data (companies, employees, departments)
fn generate_organizational_data<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
    let mut quads = Vec::with_capacity(count);
    let first_names = [
        "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    ];
    let last_names = [
        "Anderson", "Brown", "Clark", "Davis", "Evans", "Foster", "Green", "Harris",
    ];
    let departments = [
        "Engineering",
        "Sales",
        "Marketing",
        "HR",
        "Finance",
        "Operations",
    ];
    let positions = [
        "Manager",
        "Director",
        "Engineer",
        "Analyst",
        "Specialist",
        "Coordinator",
    ];
    let num_companies = ((count as f64 * 0.1) as usize).max(1);
    let num_departments = (count as f64 * 0.15) as usize;
    let num_employees = count - (num_companies + num_departments) * 2;
    for i in 0..num_companies {
        let company_uri = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/company/{}", i)).expect("Valid IRI"),
        );
        quads.push(Quad::new(
            company_uri.clone(),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("Valid IRI"),
            Term::NamedNode(NamedNode::new("http://schema.org/Organization").expect("Valid IRI")),
            GraphName::DefaultGraph,
        ));
        quads.push(Quad::new(
            company_uri,
            NamedNode::new("http://schema.org/name").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(format!("Company {}", i))),
            GraphName::DefaultGraph,
        ));
    }
    for i in 0..num_departments {
        let dept_uri = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/department/{}", i)).expect("Valid IRI"),
        );
        let dept_name = departments[rng.random_range(0..departments.len())];
        quads.push(Quad::new(
            dept_uri.clone(),
            NamedNode::new("http://schema.org/name").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(dept_name)),
            GraphName::DefaultGraph,
        ));
        let company_id = rng.random_range(0..num_companies);
        quads.push(Quad::new(
            dept_uri,
            NamedNode::new("http://schema.org/memberOf").expect("Valid IRI"),
            Term::NamedNode(
                NamedNode::new(format!("http://example.org/company/{}", company_id))
                    .expect("Valid IRI"),
            ),
            GraphName::DefaultGraph,
        ));
    }
    for i in 0..num_employees {
        let emp_uri = Subject::NamedNode(
            NamedNode::new(format!("http://example.org/employee/{}", i)).expect("Valid IRI"),
        );
        quads.push(Quad::new(
            emp_uri.clone(),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("Valid IRI"),
            Term::NamedNode(NamedNode::new("http://schema.org/Person").expect("Valid IRI")),
            GraphName::DefaultGraph,
        ));
        let first = first_names[rng.random_range(0..first_names.len())];
        let last = last_names[rng.random_range(0..last_names.len())];
        quads.push(Quad::new(
            emp_uri.clone(),
            NamedNode::new("http://schema.org/name").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(format!("{} {}", first, last))),
            GraphName::DefaultGraph,
        ));
        let position = positions[rng.random_range(0..positions.len())];
        quads.push(Quad::new(
            emp_uri.clone(),
            NamedNode::new("http://schema.org/jobTitle").expect("Valid IRI"),
            Term::Literal(Literal::new_simple_literal(position)),
            GraphName::DefaultGraph,
        ));
        if num_departments > 0 {
            let dept_id = rng.random_range(0..num_departments);
            quads.push(Quad::new(
                emp_uri,
                NamedNode::new("http://schema.org/worksFor").expect("Valid IRI"),
                Term::NamedNode(
                    NamedNode::new(format!("http://example.org/department/{}", dept_id))
                        .expect("Valid IRI"),
                ),
                GraphName::DefaultGraph,
            ));
        }
    }
    quads
}
/// Parse RDF format string into RdfFormat enum
fn parse_rdf_format(format: &str) -> Result<RdfFormat, Box<dyn std::error::Error>> {
    match format.to_lowercase().as_str() {
        "turtle" | "ttl" => Ok(RdfFormat::Turtle),
        "ntriples" | "nt" => Ok(RdfFormat::NTriples),
        "nquads" | "nq" => Ok(RdfFormat::NQuads),
        "trig" => Ok(RdfFormat::TriG),
        "rdfxml" | "rdf" | "xml" => Ok(RdfFormat::RdfXml),
        "jsonld" | "json-ld" | "json" => Ok(RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        }),
        "n3" => Ok(RdfFormat::N3),
        _ => Err(format!("Unsupported RDF format: {}", format).into()),
    }
}
/// Generate RDF data conforming to SHACL shapes
pub async fn from_shacl(
    shapes_file: PathBuf,
    output: PathBuf,
    count: usize,
    format: String,
    seed: Option<u64>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info("Generating RDF data from SHACL shapes");
    ctx.info(&format!("Shapes file: {}", shapes_file.display()));
    ctx.info(&format!("Output file: {}", output.display()));
    ctx.info(&format!("Instance count: {}", count));
    ctx.info(&format!("Format: {}", format));
    if !shapes_file.exists() {
        return Err(format!("Shapes file '{}' does not exist", shapes_file.display()).into());
    }
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let progress = helpers::query_progress();
    progress.set_message("Parsing SHACL shapes");
    let shapes = parse_shacl_shapes(&shapes_file, &ctx)?;
    ctx.info(&format!("Found {} shape definitions", shapes.len()));
    progress.set_message("Generating conforming data");
    let quads = generate_from_shapes(&mut rng, &shapes, count)?;
    ctx.info(&format!("Generated {} quads", quads.len()));
    progress.set_message("Writing output file");
    let rdf_format = parse_rdf_format(&format)?;
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("sh", "http://www.w3.org/ns/shacl#")
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("ex", "http://example.org/")
        .with_prefix("foaf", "http://xmlns.com/foaf/0.1/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        serializer.serialize_quad(quad.as_ref())?;
        written += 1;
    }
    serializer.finish()?;
    progress.finish_with_message("Generation complete");
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "✓ Dataset generated in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Quads generated: {}",
        format_number(written as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));
    if duration.as_secs_f64() > 0.0 {
        let rate = written as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Generation rate: {} quads/second",
            format_number(rate as u64)
        ));
    }
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}
/// Parse SHACL shapes from file (simplified implementation)
fn parse_shacl_shapes(
    _path: &PathBuf,
    ctx: &CliContext,
) -> Result<Vec<ShaclShape>, Box<dyn std::error::Error>> {
    ctx.info("Note: Using built-in sample SHACL shapes (full parser integration pending)");
    let person_shape = ShaclShape {
        target_class: Some("http://xmlns.com/foaf/0.1/Person".to_string()),
        properties: vec![
            PropertyConstraint {
                path: "http://xmlns.com/foaf/0.1/name".to_string(),
                min_count: Some(1),
                max_count: Some(1),
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                pattern: None,
                min_length: Some(1),
                max_length: Some(100),
                min_inclusive: None,
                max_inclusive: None,
                node_kind: Some("Literal".to_string()),
                class: None,
            },
            PropertyConstraint {
                path: "http://xmlns.com/foaf/0.1/age".to_string(),
                min_count: Some(0),
                max_count: Some(1),
                datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                pattern: None,
                min_length: None,
                max_length: None,
                min_inclusive: Some("0".to_string()),
                max_inclusive: Some("150".to_string()),
                node_kind: Some("Literal".to_string()),
                class: None,
            },
            PropertyConstraint {
                path: "http://xmlns.com/foaf/0.1/email".to_string(),
                min_count: Some(1),
                max_count: None,
                datatype: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
                pattern: Some("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$".to_string()),
                min_length: None,
                max_length: None,
                min_inclusive: None,
                max_inclusive: None,
                node_kind: Some("Literal".to_string()),
                class: None,
            },
        ],
    };
    Ok(vec![person_shape])
}
/// Generate RDF data conforming to SHACL shapes
fn generate_from_shapes<R: Rng>(
    rng: &mut R,
    shapes: &[ShaclShape],
    count: usize,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();
    for i in 0..count {
        for shape in shapes {
            let instance_uri = if let Some(target_class) = &shape.target_class {
                let class_name = target_class.split('/').next_back().unwrap_or("instance");
                Subject::NamedNode(
                    NamedNode::new(format!("http://example.org/{}/{}", class_name, i))
                        .expect("Valid IRI"),
                )
            } else {
                Subject::NamedNode(
                    NamedNode::new(format!("http://example.org/instance/{}", i))
                        .expect("Valid IRI"),
                )
            };
            if let Some(target_class) = &shape.target_class {
                quads.push(Quad::new(
                    instance_uri.clone(),
                    NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                        .expect("Valid IRI"),
                    Term::NamedNode(NamedNode::new(target_class).expect("Valid IRI")),
                    GraphName::DefaultGraph,
                ));
            }
            for prop in &shape.properties {
                let min = prop.min_count.unwrap_or(0) as usize;
                let max = prop.max_count.unwrap_or(3) as usize;
                let num_values = if max == min {
                    min
                } else {
                    rng.random_range(min..=max.min(min + 5))
                };
                for _ in 0..num_values {
                    let value = generate_property_value(rng, prop)?;
                    quads.push(Quad::new(
                        instance_uri.clone(),
                        NamedNode::new(&prop.path).expect("Valid IRI"),
                        value,
                        GraphName::DefaultGraph,
                    ));
                }
            }
        }
    }
    Ok(quads)
}
/// Generate a property value conforming to constraints
fn generate_property_value<R: Rng>(
    rng: &mut R,
    constraint: &PropertyConstraint,
) -> Result<Term, Box<dyn std::error::Error>> {
    let node_kind = constraint.node_kind.as_deref().unwrap_or("Literal");
    match node_kind {
        "IRI" | "NamedNode" => {
            if let Some(class_iri) = &constraint.class {
                Ok(Term::NamedNode(
                    NamedNode::new(class_iri).expect("Valid IRI"),
                ))
            } else {
                let id = rng.random_range(0..1000);
                Ok(Term::NamedNode(
                    NamedNode::new(format!("http://example.org/resource/{}", id))
                        .expect("Valid IRI"),
                ))
            }
        }
        "Literal" => {
            let datatype = constraint
                .datatype
                .as_deref()
                .unwrap_or("http://www.w3.org/2001/XMLSchema#string");
            let value_str = match datatype {
                "http://www.w3.org/2001/XMLSchema#string" => generate_string_value(rng, constraint),
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int" => generate_integer_value(rng, constraint),
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double" => {
                    generate_decimal_value(rng, constraint)
                }
                "http://www.w3.org/2001/XMLSchema#boolean" => if rng.random_range(0..2) == 0 {
                    "true"
                } else {
                    "false"
                }
                .to_string(),
                "http://www.w3.org/2001/XMLSchema#date" => {
                    let year = rng.random_range(1900..=2024);
                    let month = rng.random_range(1..=12);
                    let day = rng.random_range(1..=28);
                    format!("{:04}-{:02}-{:02}", year, month, day)
                }
                "http://www.w3.org/2001/XMLSchema#dateTime" => {
                    let year = rng.random_range(2000..=2024);
                    let month = rng.random_range(1..=12);
                    let day = rng.random_range(1..=28);
                    let hour = rng.random_range(0..=23);
                    let minute = rng.random_range(0..=59);
                    let second = rng.random_range(0..=59);
                    format!(
                        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                        year, month, day, hour, minute, second
                    )
                }
                _ => generate_string_value(rng, constraint),
            };
            Ok(Term::Literal(Literal::new_typed_literal(
                value_str,
                NamedNode::new(datatype).expect("Valid IRI"),
            )))
        }
        _ => Ok(Term::Literal(Literal::new_simple_literal(
            generate_string_value(rng, constraint),
        ))),
    }
}
/// Generate string value conforming to constraints
fn generate_string_value<R: Rng>(rng: &mut R, constraint: &PropertyConstraint) -> String {
    if let Some(pattern) = &constraint.pattern {
        if pattern.contains("@") {
            let names = ["alice", "bob", "carol", "dave", "emma"];
            let domains = ["example.com", "test.org", "demo.net"];
            return format!(
                "{}{}@{}",
                names[rng.random_range(0..names.len())],
                rng.random_range(1..100),
                domains[rng.random_range(0..domains.len())]
            );
        }
    }
    let min_len = constraint.min_length.unwrap_or(1) as usize;
    let max_len = constraint.max_length.unwrap_or(50) as usize;
    let target_len = rng.random_range(min_len..=max_len);
    let words = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    ];
    let mut result = String::new();
    while result.len() < target_len {
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(words[rng.random_range(0..words.len())]);
    }
    result.truncate(max_len);
    result
}
/// Generate integer value conforming to constraints
fn generate_integer_value<R: Rng>(rng: &mut R, constraint: &PropertyConstraint) -> String {
    let min = constraint
        .min_inclusive
        .as_ref()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0);
    let max = constraint
        .max_inclusive
        .as_ref()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(1000);
    rng.random_range(min..=max).to_string()
}
/// Generate decimal value conforming to constraints
fn generate_decimal_value<R: Rng>(rng: &mut R, constraint: &PropertyConstraint) -> String {
    let min = constraint
        .min_inclusive
        .as_ref()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let max = constraint
        .max_inclusive
        .as_ref()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1000.0);
    let value = min + (max - min) * (rng.random_range(0..10000) as f64 / 10000.0);
    format!("{:.2}", value)
}
/// Generate RDF data conforming to RDFS schema
pub async fn from_rdfs(
    schema_file: PathBuf,
    output: PathBuf,
    count: usize,
    format: String,
    seed: Option<u64>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info("Generating RDF data from RDFS schema");
    ctx.info(&format!("Schema file: {}", schema_file.display()));
    ctx.info(&format!("Output file: {}", output.display()));
    ctx.info(&format!("Instance count: {}", count));
    ctx.info(&format!("Format: {}", format));
    if !schema_file.exists() {
        return Err(format!("Schema file '{}' does not exist", schema_file.display()).into());
    }
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let progress = helpers::query_progress();
    progress.set_message("Parsing RDFS schema");
    let schema = parse_rdfs_schema(&schema_file, &ctx)?;
    ctx.info(&format!(
        "Found {} classes and {} properties",
        schema.classes.len(),
        schema.properties.len()
    ));
    progress.set_message("Generating conforming data");
    let quads = generate_from_rdfs_schema(&mut rng, &schema, count)?;
    ctx.info(&format!("Generated {} quads", quads.len()));
    progress.set_message("Writing output file");
    let rdf_format = parse_rdf_format(&format)?;
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("ex", "http://example.org/")
        .with_prefix("foaf", "http://xmlns.com/foaf/0.1/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        serializer.serialize_quad(quad.as_ref())?;
        written += 1;
    }
    serializer.finish()?;
    progress.finish_with_message("Generation complete");
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "✓ Dataset generated in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Quads generated: {}",
        format_number(written as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));
    if duration.as_secs_f64() > 0.0 {
        let rate = written as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Generation rate: {} quads/second",
            format_number(rate as u64)
        ));
    }
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}
/// Parse RDFS schema from file (simplified implementation)
fn parse_rdfs_schema(
    _path: &PathBuf,
    ctx: &CliContext,
) -> Result<RdfsSchema, Box<dyn std::error::Error>> {
    ctx.info("Note: Using built-in sample RDFS schema (full parser integration pending)");
    let classes = vec![
        RdfsClass {
            uri: "http://example.org/Person".to_string(),
            _label: Some("Person".to_string()),
            _comment: Some("A human being".to_string()),
            _super_classes: vec!["http://www.w3.org/2000/01/rdf-schema#Resource".to_string()],
        },
        RdfsClass {
            uri: "http://example.org/Organization".to_string(),
            _label: Some("Organization".to_string()),
            _comment: Some("An organized group of people".to_string()),
            _super_classes: vec!["http://www.w3.org/2000/01/rdf-schema#Resource".to_string()],
        },
        RdfsClass {
            uri: "http://example.org/Document".to_string(),
            _label: Some("Document".to_string()),
            _comment: Some("A written or digital document".to_string()),
            _super_classes: vec!["http://www.w3.org/2000/01/rdf-schema#Resource".to_string()],
        },
    ];
    let properties = vec![
        RdfsProperty {
            uri: "http://example.org/name".to_string(),
            _label: Some("name".to_string()),
            _comment: Some("The name of something".to_string()),
            domain: vec![
                "http://example.org/Person".to_string(),
                "http://example.org/Organization".to_string(),
            ],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec!["http://www.w3.org/2000/01/rdf-schema#label".to_string()],
        },
        RdfsProperty {
            uri: "http://example.org/age".to_string(),
            _label: Some("age".to_string()),
            _comment: Some("The age of a person".to_string()),
            domain: vec!["http://example.org/Person".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#integer".to_string()],
            _super_properties: vec![],
        },
        RdfsProperty {
            uri: "http://example.org/email".to_string(),
            _label: Some("email".to_string()),
            _comment: Some("Email address".to_string()),
            domain: vec!["http://example.org/Person".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
        },
        RdfsProperty {
            uri: "http://example.org/employedBy".to_string(),
            _label: Some("employed by".to_string()),
            _comment: Some("The organization that employs a person".to_string()),
            domain: vec!["http://example.org/Person".to_string()],
            range: vec!["http://example.org/Organization".to_string()],
            _super_properties: vec![],
        },
        RdfsProperty {
            uri: "http://example.org/foundedYear".to_string(),
            _label: Some("founded year".to_string()),
            _comment: Some("The year an organization was founded".to_string()),
            domain: vec!["http://example.org/Organization".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#integer".to_string()],
            _super_properties: vec![],
        },
        RdfsProperty {
            uri: "http://example.org/author".to_string(),
            _label: Some("author".to_string()),
            _comment: Some("The author of a document".to_string()),
            domain: vec!["http://example.org/Document".to_string()],
            range: vec!["http://example.org/Person".to_string()],
            _super_properties: vec!["http://purl.org/dc/terms/creator".to_string()],
        },
        RdfsProperty {
            uri: "http://example.org/title".to_string(),
            _label: Some("title".to_string()),
            _comment: Some("The title of a document".to_string()),
            domain: vec!["http://example.org/Document".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
        },
    ];
    Ok(RdfsSchema {
        classes,
        properties,
    })
}
/// Generate RDF data conforming to RDFS schema
fn generate_from_rdfs_schema<R: Rng>(
    rng: &mut R,
    schema: &RdfsSchema,
    count: usize,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();
    let instances_per_class = if !schema.classes.is_empty() {
        (count as f64 / schema.classes.len() as f64).ceil() as usize
    } else {
        return Ok(quads);
    };
    let mut generated_instances: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for class in &schema.classes {
        let mut class_instances = Vec::new();
        for i in 0..instances_per_class {
            let class_name = class.uri.split('/').next_back().unwrap_or("instance");
            let instance_uri_str = format!("http://example.org/{}/{}", class_name, i);
            let instance_uri =
                Subject::NamedNode(NamedNode::new(&instance_uri_str).expect("Valid IRI"));
            class_instances.push(instance_uri_str.clone());
            quads.push(Quad::new(
                instance_uri.clone(),
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .expect("Valid IRI"),
                Term::NamedNode(NamedNode::new(&class.uri).expect("Valid IRI")),
                GraphName::DefaultGraph,
            ));
            for property in &schema.properties {
                if property.domain.contains(&class.uri) {
                    let value =
                        generate_rdfs_property_value(rng, property, &generated_instances, schema)?;
                    quads.push(Quad::new(
                        instance_uri.clone(),
                        NamedNode::new(&property.uri).expect("Valid IRI"),
                        value,
                        GraphName::DefaultGraph,
                    ));
                }
            }
        }
        generated_instances.insert(class.uri.clone(), class_instances);
    }
    Ok(quads)
}
/// Generate a property value based on RDFS range constraints
fn generate_rdfs_property_value<R: Rng>(
    rng: &mut R,
    property: &RdfsProperty,
    generated_instances: &std::collections::HashMap<String, Vec<String>>,
    schema: &RdfsSchema,
) -> Result<Term, Box<dyn std::error::Error>> {
    let range_uri = property.range.first().ok_or("Property has no range")?;
    if schema.classes.iter().any(|c| &c.uri == range_uri) {
        if let Some(instances) = generated_instances.get(range_uri) {
            if !instances.is_empty() {
                let idx = rng.random_range(0..instances.len());
                return Ok(Term::NamedNode(
                    NamedNode::new(&instances[idx]).expect("Valid IRI"),
                ));
            }
        }
        let class_name = range_uri.split('/').next_back().unwrap_or("resource");
        return Ok(Term::NamedNode(
            NamedNode::new(format!(
                "http://example.org/{}/{}",
                class_name,
                rng.random_range(0..1000)
            ))
            .expect("Valid IRI"),
        ));
    }
    let value_str = match range_uri.as_str() {
        "http://www.w3.org/2001/XMLSchema#string" => {
            let words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"];
            let mut result = String::new();
            for _ in 0..rng.random_range(1..4) {
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(words[rng.random_range(0..words.len())]);
            }
            result
        }
        "http://www.w3.org/2001/XMLSchema#integer" | "http://www.w3.org/2001/XMLSchema#int" => {
            if property.uri.contains("age") {
                rng.random_range(18..80).to_string()
            } else if property.uri.contains("year") || property.uri.contains("Year") {
                rng.random_range(1950..2024).to_string()
            } else {
                rng.random_range(0..1000).to_string()
            }
        }
        "http://www.w3.org/2001/XMLSchema#decimal" | "http://www.w3.org/2001/XMLSchema#double" => {
            let value = rng.random_range(0..10000) as f64 / 100.0;
            format!("{:.2}", value)
        }
        "http://www.w3.org/2001/XMLSchema#boolean" => if rng.random_range(0..2) == 0 {
            "true"
        } else {
            "false"
        }
        .to_string(),
        "http://www.w3.org/2001/XMLSchema#date" => {
            let year = rng.random_range(1950..=2024);
            let month = rng.random_range(1..=12);
            let day = rng.random_range(1..=28);
            format!("{:04}-{:02}-{:02}", year, month, day)
        }
        "http://www.w3.org/2001/XMLSchema#dateTime" => {
            let year = rng.random_range(2000..=2024);
            let month = rng.random_range(1..=12);
            let day = rng.random_range(1..=28);
            let hour = rng.random_range(0..=23);
            let minute = rng.random_range(0..=59);
            let second = rng.random_range(0..=59);
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                year, month, day, hour, minute, second
            )
        }
        _ => format!("value_{}", rng.random_range(0..1000)),
    };
    Ok(Term::Literal(Literal::new_typed_literal(
        value_str,
        NamedNode::new(range_uri).expect("Valid IRI"),
    )))
}
/// Generate RDF data conforming to OWL ontology
pub async fn from_owl(
    ontology_file: PathBuf,
    output: PathBuf,
    count: usize,
    format: String,
    seed: Option<u64>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info("Generating RDF data from OWL ontology");
    ctx.info(&format!("Ontology file: {}", ontology_file.display()));
    ctx.info(&format!("Output file: {}", output.display()));
    ctx.info(&format!("Instance count: {}", count));
    ctx.info(&format!("Format: {}", format));
    if !ontology_file.exists() {
        return Err(format!("Ontology file '{}' does not exist", ontology_file.display()).into());
    }
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let progress = helpers::query_progress();
    progress.set_message("Parsing OWL ontology");
    let ontology = parse_owl_ontology(&ontology_file, &ctx)?;
    ctx.info(&format!(
        "Found {} classes and {} properties",
        ontology.classes.len(),
        ontology.properties.len()
    ));
    progress.set_message("Generating conforming data");
    let quads = generate_from_owl_ontology(&mut rng, &ontology, count)?;
    ctx.info(&format!("Generated {} quads", quads.len()));
    progress.set_message("Writing output file");
    let rdf_format = parse_rdf_format(&format)?;
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("owl", "http://www.w3.org/2002/07/owl#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("ex", "http://example.org/")
        .with_prefix("foaf", "http://xmlns.com/foaf/0.1/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        serializer.serialize_quad(quad.as_ref())?;
        written += 1;
    }
    serializer.finish()?;
    progress.finish_with_message("Generation complete");
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "✓ Dataset generated in {}",
        format_duration(duration)
    ));
    ctx.info(&format!(
        "  Quads generated: {}",
        format_number(written as u64)
    ));
    ctx.info(&format!("  File size: {}", format_bytes(file_size)));
    if duration.as_secs_f64() > 0.0 {
        let rate = written as f64 / duration.as_secs_f64();
        ctx.info(&format!(
            "  Generation rate: {} quads/second",
            format_number(rate as u64)
        ));
    }
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}
/// Parse OWL ontology from file (simplified implementation)
fn parse_owl_ontology(
    _path: &PathBuf,
    ctx: &CliContext,
) -> Result<OwlOntology, Box<dyn std::error::Error>> {
    ctx.info("Note: Using built-in sample OWL ontology (full parser integration pending)");
    let classes = vec![
        OwlClass {
            uri: "http://example.org/University".to_string(),
            _label: Some("University".to_string()),
            _comment: Some("An institution of higher education".to_string()),
            _super_classes: vec!["http://example.org/Organization".to_string()],
            _equivalent_classes: vec![],
            _disjoint_with: vec!["http://example.org/Person".to_string()],
            restrictions: vec![
                OwlRestriction {
                    on_property: "http://example.org/hasStudent".to_string(),
                    restriction_type: OwlRestrictionType::MinCardinality(10),
                },
                OwlRestriction {
                    on_property: "http://example.org/hasFaculty".to_string(),
                    restriction_type: OwlRestrictionType::MinCardinality(5),
                },
            ],
        },
        OwlClass {
            uri: "http://example.org/Professor".to_string(),
            _label: Some("Professor".to_string()),
            _comment: Some("A university faculty member".to_string()),
            _super_classes: vec!["http://example.org/Person".to_string()],
            _equivalent_classes: vec![],
            _disjoint_with: vec!["http://example.org/Student".to_string()],
            restrictions: vec![
                OwlRestriction {
                    on_property: "http://example.org/teachesAt".to_string(),
                    restriction_type: OwlRestrictionType::ExactCardinality(1),
                },
                OwlRestriction {
                    on_property: "http://example.org/officeNumber".to_string(),
                    restriction_type: OwlRestrictionType::ExactCardinality(1),
                },
            ],
        },
        OwlClass {
            uri: "http://example.org/Student".to_string(),
            _label: Some("Student".to_string()),
            _comment: Some("A person enrolled in a university".to_string()),
            _super_classes: vec!["http://example.org/Person".to_string()],
            _equivalent_classes: vec![],
            _disjoint_with: vec!["http://example.org/Professor".to_string()],
            restrictions: vec![
                OwlRestriction {
                    on_property: "http://example.org/enrolledIn".to_string(),
                    restriction_type: OwlRestrictionType::ExactCardinality(1),
                },
                OwlRestriction {
                    on_property: "http://example.org/studentID".to_string(),
                    restriction_type: OwlRestrictionType::ExactCardinality(1),
                },
            ],
        },
        OwlClass {
            uri: "http://example.org/Course".to_string(),
            _label: Some("Course".to_string()),
            _comment: Some("An academic course".to_string()),
            _super_classes: vec![],
            _equivalent_classes: vec![],
            _disjoint_with: vec![],
            restrictions: vec![
                OwlRestriction {
                    on_property: "http://example.org/taughtBy".to_string(),
                    restriction_type: OwlRestrictionType::MinCardinality(1),
                },
                OwlRestriction {
                    on_property: "http://example.org/taughtBy".to_string(),
                    restriction_type: OwlRestrictionType::MaxCardinality(2),
                },
            ],
        },
    ];
    let properties = vec![
        OwlProperty {
            uri: "http://example.org/teachesAt".to_string(),
            _label: Some("teaches at".to_string()),
            _comment: Some("University where a professor teaches".to_string()),
            property_type: OwlPropertyType::Object,
            domain: vec!["http://example.org/Professor".to_string()],
            range: vec!["http://example.org/University".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/enrolledIn".to_string(),
            _label: Some("enrolled in".to_string()),
            _comment: Some("University where a student is enrolled".to_string()),
            property_type: OwlPropertyType::Object,
            domain: vec!["http://example.org/Student".to_string()],
            range: vec!["http://example.org/University".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/studentID".to_string(),
            _label: Some("student ID".to_string()),
            _comment: Some("Unique identifier for a student".to_string()),
            property_type: OwlPropertyType::Datatype,
            domain: vec!["http://example.org/Student".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: true,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/officeNumber".to_string(),
            _label: Some("office number".to_string()),
            _comment: Some("Office number of a professor".to_string()),
            property_type: OwlPropertyType::Datatype,
            domain: vec!["http://example.org/Professor".to_string()],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/taughtBy".to_string(),
            _label: Some("taught by".to_string()),
            _comment: Some("Professor who teaches a course".to_string()),
            property_type: OwlPropertyType::Object,
            domain: vec!["http://example.org/Course".to_string()],
            range: vec!["http://example.org/Professor".to_string()],
            _super_properties: vec![],
            is_functional: false,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
        OwlProperty {
            uri: "http://example.org/hasColleague".to_string(),
            _label: Some("has colleague".to_string()),
            _comment: Some("Colleague relationship between professors".to_string()),
            property_type: OwlPropertyType::Object,
            domain: vec!["http://example.org/Professor".to_string()],
            range: vec!["http://example.org/Professor".to_string()],
            _super_properties: vec![],
            is_functional: false,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: true,
        },
        OwlProperty {
            uri: "http://example.org/name".to_string(),
            _label: Some("name".to_string()),
            _comment: Some("Name of a person or organization".to_string()),
            property_type: OwlPropertyType::Datatype,
            domain: vec![
                "http://example.org/Person".to_string(),
                "http://example.org/University".to_string(),
            ],
            range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
            _super_properties: vec![],
            is_functional: true,
            is_inverse_functional: false,
            _is_transitive: false,
            is_symmetric: false,
        },
    ];
    Ok(OwlOntology {
        classes,
        properties,
    })
}
/// Generate RDF data conforming to OWL ontology
fn generate_from_owl_ontology<R: Rng>(
    rng: &mut R,
    ontology: &OwlOntology,
    count: usize,
) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let mut quads = Vec::new();
    let instances_per_class = if !ontology.classes.is_empty() {
        (count as f64 / ontology.classes.len() as f64).ceil() as usize
    } else {
        return Ok(quads);
    };
    let mut generated_instances: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for class in &ontology.classes {
        let mut class_instances = Vec::new();
        for i in 0..instances_per_class {
            let class_name = class.uri.split('/').next_back().unwrap_or("instance");
            let instance_uri_str = format!("http://example.org/{}/{}", class_name, i);
            let instance_uri =
                Subject::NamedNode(NamedNode::new(&instance_uri_str).expect("Valid IRI"));
            class_instances.push(instance_uri_str.clone());
            quads.push(Quad::new(
                instance_uri.clone(),
                NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .expect("Valid IRI"),
                Term::NamedNode(NamedNode::new(&class.uri).expect("Valid IRI")),
                GraphName::DefaultGraph,
            ));
            for property in &ontology.properties {
                if property.domain.contains(&class.uri) {
                    let cardinality = get_property_cardinality(class, &property.uri);
                    let num_values = if property.is_functional {
                        1
                    } else {
                        match cardinality {
                            Some((min, Some(max))) => rng.random_range(min..=max),
                            Some((min, None)) => rng.random_range(min..=(min + 3)),
                            None => rng.random_range(1..=2),
                        }
                    };
                    let mut used_values = std::collections::HashSet::new();
                    for _ in 0..num_values {
                        let value = generate_owl_property_value(
                            rng,
                            property,
                            &generated_instances,
                            ontology,
                            &mut used_values,
                        )?;
                        quads.push(Quad::new(
                            instance_uri.clone(),
                            NamedNode::new(&property.uri).expect("Valid IRI"),
                            value,
                            GraphName::DefaultGraph,
                        ));
                    }
                    if property.is_symmetric {}
                }
            }
        }
        generated_instances.insert(class.uri.clone(), class_instances);
    }
    Ok(quads)
}
/// Get cardinality constraints from OWL restrictions
fn get_property_cardinality(
    class: &OwlClass,
    property_uri: &str,
) -> Option<(usize, Option<usize>)> {
    let mut min_card: Option<usize> = None;
    let mut max_card: Option<usize> = None;
    for restriction in &class.restrictions {
        if restriction.on_property == property_uri {
            match &restriction.restriction_type {
                OwlRestrictionType::MinCardinality(n) => min_card = Some(*n as usize),
                OwlRestrictionType::MaxCardinality(n) => max_card = Some(*n as usize),
                OwlRestrictionType::ExactCardinality(n) => {
                    min_card = Some(*n as usize);
                    max_card = Some(*n as usize);
                }
                _ => {}
            }
        }
    }
    if min_card.is_some() || max_card.is_some() {
        Some((min_card.unwrap_or(0), max_card))
    } else {
        None
    }
}
/// Generate a property value based on OWL property characteristics
fn generate_owl_property_value<R: Rng>(
    rng: &mut R,
    property: &OwlProperty,
    generated_instances: &std::collections::HashMap<String, Vec<String>>,
    ontology: &OwlOntology,
    used_values: &mut std::collections::HashSet<String>,
) -> Result<Term, Box<dyn std::error::Error>> {
    match property.property_type {
        OwlPropertyType::Object => {
            let range_uri = property.range.first().ok_or("Property has no range")?;
            if ontology.classes.iter().any(|c| &c.uri == range_uri) {
                if let Some(instances) = generated_instances.get(range_uri) {
                    if !instances.is_empty() {
                        let idx = rng.random_range(0..instances.len());
                        return Ok(Term::NamedNode(
                            NamedNode::new(&instances[idx]).expect("Valid IRI"),
                        ));
                    }
                }
            }
            let class_name = range_uri.split('/').next_back().unwrap_or("resource");
            Ok(Term::NamedNode(
                NamedNode::new(format!(
                    "http://example.org/{}/{}",
                    class_name,
                    rng.random_range(0..1000)
                ))
                .expect("Valid IRI"),
            ))
        }
        OwlPropertyType::Datatype => {
            let range_uri = property.range.first().ok_or("Property has no range")?;
            let mut value_str = generate_datatype_value(rng, range_uri, &property.uri)?;
            if property.is_inverse_functional {
                let mut attempts = 0;
                while used_values.contains(&value_str) && attempts < 100 {
                    value_str = generate_datatype_value(rng, range_uri, &property.uri)?;
                    attempts += 1;
                }
                used_values.insert(value_str.clone());
            }
            Ok(Term::Literal(Literal::new_typed_literal(
                value_str,
                NamedNode::new(range_uri).expect("Valid IRI"),
            )))
        }
    }
}
/// Generate a datatype value based on range and property hints
fn generate_datatype_value<R: Rng>(
    rng: &mut R,
    range_uri: &str,
    property_uri: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let value_str = match range_uri {
        "http://www.w3.org/2001/XMLSchema#string" => {
            if property_uri.contains("studentID") {
                format!("STU{:06}", rng.random_range(100000..999999))
            } else if property_uri.contains("officeNumber") {
                format!("Room-{:03}", rng.random_range(100..999))
            } else if property_uri.contains("name") {
                let first_names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"];
                let last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"];
                format!(
                    "{} {}",
                    first_names[rng.random_range(0..first_names.len())],
                    last_names[rng.random_range(0..last_names.len())]
                )
            } else {
                let words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"];
                let mut result = String::new();
                for _ in 0..rng.random_range(1..3) {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(words[rng.random_range(0..words.len())]);
                }
                result
            }
        }
        "http://www.w3.org/2001/XMLSchema#integer" | "http://www.w3.org/2001/XMLSchema#int" => {
            rng.random_range(1..1000).to_string()
        }
        "http://www.w3.org/2001/XMLSchema#decimal" | "http://www.w3.org/2001/XMLSchema#double" => {
            let value = rng.random_range(0..10000) as f64 / 100.0;
            format!("{:.2}", value)
        }
        "http://www.w3.org/2001/XMLSchema#boolean" => if rng.random_range(0..2) == 0 {
            "true"
        } else {
            "false"
        }
        .to_string(),
        _ => format!("value_{}", rng.random_range(0..1000)),
    };
    Ok(value_str)
}
/// Detect the type of schema file (SHACL, RDFS, or OWL)
fn detect_schema_type(
    schema_file: &PathBuf,
    ctx: &CliContext,
) -> Result<String, Box<dyn std::error::Error>> {
    use std::io::BufReader;
    let format = match schema_file
        .extension()
        .and_then(|e| e.to_str())
        .ok_or("File has no extension")?
        .to_lowercase()
        .as_str()
    {
        "ttl" => RdfFormat::Turtle,
        "rdf" | "xml" => RdfFormat::RdfXml,
        "nt" => RdfFormat::NTriples,
        "nq" => RdfFormat::NQuads,
        "trig" => RdfFormat::TriG,
        "jsonld" | "json" => RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        },
        "n3" => RdfFormat::N3,
        ext => return Err(format!("Unsupported file extension: {}", ext).into()),
    };
    let file = fs::File::open(schema_file)?;
    let reader = BufReader::new(file);
    let parser = RdfParser::new(format);
    let mut has_shacl = false;
    let mut has_rdfs = false;
    let mut has_owl = false;
    let shacl_ns = "http://www.w3.org/ns/shacl#";
    let rdfs_ns = "http://www.w3.org/2000/01/rdf-schema#";
    let owl_ns = "http://www.w3.org/2002/07/owl#";
    let mut quad_count = 0;
    for quad in parser.for_reader(reader).flatten() {
        quad_count += 1;
        let pred_uri = quad.predicate().as_str();
        if pred_uri.starts_with(shacl_ns) {
            has_shacl = true;
        }
        if pred_uri.starts_with(rdfs_ns) {
            has_rdfs = true;
        }
        if pred_uri.starts_with(owl_ns) {
            has_owl = true;
        }
        if let oxirs_core::model::Object::NamedNode(nn) = quad.object() {
            let obj_uri = nn.as_str();
            if obj_uri.starts_with(shacl_ns) {
                has_shacl = true;
            }
            if obj_uri.starts_with(rdfs_ns) {
                has_rdfs = true;
            }
            if obj_uri.starts_with(owl_ns) {
                has_owl = true;
            }
        }
        if quad_count > 100 {
            break;
        }
    }
    ctx.info(&format!(
        "Schema detection: SHACL={}, RDFS={}, OWL={}",
        has_shacl, has_rdfs, has_owl
    ));
    if has_shacl {
        Ok("SHACL".to_string())
    } else if has_owl {
        Ok("OWL".to_string())
    } else if has_rdfs {
        Ok("RDFS".to_string())
    } else {
        Err(
            "Unable to detect schema type. File must contain SHACL, RDFS, or OWL definitions."
                .into(),
        )
    }
}
/// Run schema-based RDF data generation (SHACL/RDFS/OWL)
async fn run_schema_based_generation(
    output: PathBuf,
    size: String,
    schema_file: PathBuf,
    format: String,
    seed: Option<u64>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info("Schema-based RDF data generation");
    ctx.info(&format!("Schema file: {}", schema_file.display()));
    ctx.info(&format!("Output file: {}", output.display()));
    let size_enum = DatasetSize::from_string(&size)?;
    let instance_count = size_enum.triple_count();
    ctx.info(&format!("Instances to generate: {}", instance_count));
    ctx.info(&format!("Format: {}", format));
    if !schema_file.exists() {
        return Err(format!("Schema file '{}' not found", schema_file.display()).into());
    }
    let mut rng = if let Some(s) = seed {
        ctx.info(&format!("Random seed: {}", s));
        Random::seed(s)
    } else {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random::seed(timestamp)
    };
    if output.exists() {
        return Err(format!("Output file '{}' already exists", output.display()).into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let start_time = Instant::now();
    let mut data_logger = DataLogger::new("generate_schema", output.to_str().unwrap_or("unknown"));
    let mut perf_logger = PerfLogger::new("generate_schema");
    perf_logger.add_metadata("schema_file", schema_file.display().to_string());
    perf_logger.add_metadata("instance_count", instance_count.to_string());
    perf_logger.add_metadata("format", &format);
    if let Some(s) = seed {
        perf_logger.add_metadata("seed", s.to_string());
    }
    ctx.info("Detecting schema type...");
    let schema_type = detect_schema_type(&schema_file, &ctx)?;
    ctx.info(&format!("Detected schema type: {}", schema_type));
    let quads = match schema_type.as_str() {
        "SHACL" => {
            ctx.info("Parsing SHACL shapes...");
            let shapes = shacl::parse_shacl_shapes(&schema_file, &ctx)?;
            if shapes.is_empty() {
                return Err(
                    "No SHACL shapes found in schema file. Ensure the file contains sh:NodeShape definitions with sh:targetClass."
                        .into(),
                );
            }
            ctx.info(&format!(
                "Found {} shapes with target classes",
                shapes.len()
            ));
            ctx.info("Generating RDF data from SHACL shapes...");
            shacl::generate_from_shapes(&shapes, instance_count, &mut rng, &ctx)?
        }
        "RDFS" => {
            ctx.info("Parsing RDFS schema...");
            let schema = rdfs::parse_rdfs_schema(&schema_file, &ctx)?;
            if schema.classes.is_empty() {
                return Err(
                    "No RDFS classes found in schema file. Ensure the file contains rdfs:Class definitions."
                        .into(),
                );
            }
            ctx.info(&format!(
                "Found {} RDFS classes and {} properties",
                schema.classes.len(),
                schema.properties.len()
            ));
            ctx.info("Generating RDF data from RDFS schema...");
            rdfs::generate_from_rdfs_schema(&schema, instance_count, &mut rng, &ctx)?
        }
        "OWL" => {
            ctx.info("Parsing OWL ontology...");
            let ontology = owl::parse_owl_ontology(&schema_file, &ctx)?;

            if ontology.classes.is_empty() {
                return Err("No OWL classes found in ontology file. Ensure the file contains owl:Class definitions.".into());
            }

            ctx.info(&format!(
                "Found {} OWL classes, {} properties, {} individuals",
                ontology.classes.len(),
                ontology.properties.len(),
                ontology.individuals.len()
            ));
            ctx.info("Generating RDF data from OWL ontology...");
            owl::generate_from_owl_ontology(&ontology, instance_count, &mut rng, &ctx)?
        }
        _ => {
            return Err(format!(
                "Unknown schema type: {}. Supported types: SHACL, RDFS, OWL",
                schema_type
            )
            .into());
        }
    };
    ctx.info(&format!("Generated {} RDF quads", quads.len()));
    let rdf_format = parse_rdf_format(&format)?;
    ctx.info("Writing to file...");
    let output_file = fs::File::create(&output)?;
    let mut serializer = RdfSerializer::new(rdf_format)
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .with_prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
        .with_prefix("sh", "http://www.w3.org/ns/shacl#")
        .with_prefix("ex", "http://example.org/")
        .pretty()
        .for_writer(output_file);
    let mut written = 0;
    for quad in &quads {
        match serializer.serialize_quad(quad.as_ref()) {
            Ok(_) => written += 1,
            Err(e) => {
                return Err(format!("Failed to serialize quad: {}", e).into());
            }
        }
    }
    serializer
        .finish()
        .map_err(|e| format!("Failed to finalize serialization: {}", e))?;
    let duration = start_time.elapsed();
    let file_size = fs::metadata(&output)?.len();
    data_logger.update_progress(file_size, written);
    data_logger.complete();
    perf_logger.add_metadata("quad_count", written);
    perf_logger.complete(Some(5000));
    ctx.info("Generation Statistics");
    ctx.success(&format!(
        "Generation completed in {}",
        format_duration(duration)
    ));
    ctx.info(&format!("Quads generated: {}", format_number(written)));
    ctx.info(&format!("File size: {}", format_bytes(file_size)));
    ctx.info(&format!(
        "Generation rate: {:.0} quads/second",
        written as f64 / duration.as_secs_f64()
    ));
    ctx.success(&format!("Output written to: {}", output.display()));
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dataset_size_parsing() {
        assert!(matches!(
            DatasetSize::from_string("tiny"),
            Ok(DatasetSize::Tiny)
        ));
        assert!(matches!(
            DatasetSize::from_string("small"),
            Ok(DatasetSize::Small)
        ));
        assert!(matches!(
            DatasetSize::from_string("medium"),
            Ok(DatasetSize::Medium)
        ));
        assert!(matches!(
            DatasetSize::from_string("1000"),
            Ok(DatasetSize::Custom(1000))
        ));
        assert!(DatasetSize::from_string("invalid").is_err());
    }
    #[test]
    fn test_dataset_type_parsing() {
        assert!(matches!(
            DatasetType::from_string("rdf"),
            Ok(DatasetType::Rdf)
        ));
        assert!(matches!(
            DatasetType::from_string("graph"),
            Ok(DatasetType::Graph)
        ));
        assert!(matches!(
            DatasetType::from_string("semantic"),
            Ok(DatasetType::Semantic)
        ));
        assert!(matches!(
            DatasetType::from_string("bibliographic"),
            Ok(DatasetType::Bibliographic)
        ));
        assert!(matches!(
            DatasetType::from_string("bib"),
            Ok(DatasetType::Bibliographic)
        ));
        assert!(matches!(
            DatasetType::from_string("geographic"),
            Ok(DatasetType::Geographic)
        ));
        assert!(matches!(
            DatasetType::from_string("geo"),
            Ok(DatasetType::Geographic)
        ));
        assert!(matches!(
            DatasetType::from_string("organizational"),
            Ok(DatasetType::Organizational)
        ));
        assert!(matches!(
            DatasetType::from_string("org"),
            Ok(DatasetType::Organizational)
        ));
        assert!(DatasetType::from_string("invalid").is_err());
    }
    #[test]
    fn test_triple_counts() {
        assert_eq!(DatasetSize::Tiny.triple_count(), 100);
        assert_eq!(DatasetSize::Small.triple_count(), 1_000);
        assert_eq!(DatasetSize::Medium.triple_count(), 10_000);
        assert_eq!(DatasetSize::Large.triple_count(), 100_000);
        assert_eq!(DatasetSize::XLarge.triple_count(), 1_000_000);
        assert_eq!(DatasetSize::Custom(500).triple_count(), 500);
    }
}
