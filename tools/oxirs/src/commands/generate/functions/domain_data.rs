//! Domain-specific RDF data generation
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxirs_core::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use scirs2_core::Rng;

/// Generate semantic web data (classes, properties, instances)
pub(super) fn generate_semantic_data<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
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
pub(super) fn generate_bibliographic_data<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
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
pub(super) fn generate_geographic_data<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
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
pub(super) fn generate_organizational_data<R: Rng>(rng: &mut R, count: usize) -> Vec<Quad> {
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
