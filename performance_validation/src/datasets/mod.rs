use rand_distr::Distribution;use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use scirs2_core::random::Random;
use rand_distr::{Uniform, Normal};

// Helper function for normal distribution using Box-Muller transform
fn generate_normal(rng: &mut Random<impl rand::Rng>, mean: f64, std_dev: f64) -> f64 {
    use std::f64::consts::PI;
    static mut SPARE: Option<f64> = None;
    static mut HAS_SPARE: bool = false;

    unsafe {
        if HAS_SPARE {
            HAS_SPARE = false;
            return SPARE.unwrap() * std_dev + mean;
        }

        HAS_SPARE = true;
        let u: f64 = rng.random_range(0.0, 1.0);
        let v: f64 = rng.random_range(0.0, 1.0);
        let mag = std_dev * (-2.0 * u.ln()).sqrt();
        SPARE = Some(mag * (2.0 * PI * v).sin());
        mag * (2.0 * PI * v).cos() + mean
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub size: DatasetSize,
    pub format: DatasetFormat,
    pub complexity: DatasetComplexity,
    pub include_embeddings: bool,
    pub embedding_dimensions: usize,
    pub domain: DatasetDomain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetSize {
    Small,      // ~1K triples
    Medium,     // ~10K triples
    Large,      // ~100K triples
    XLarge,     // ~1M triples
    XXLarge,    // ~10M triples
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetFormat {
    NTriples,
    Turtle,
    RdfXml,
    JsonLd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetComplexity {
    Simple,     // Basic subject-predicate-object patterns
    Medium,     // Some complex queries, moderate joins
    Complex,    // Complex queries, multiple joins, unions
    VeryComplex, // Advanced SPARQL features, nested queries
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetDomain {
    General,
    Biomedical,
    Geographic,
    Academic,
    Cultural,
    Technology,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub config: DatasetConfig,
    pub triples: Vec<Triple>,
    pub embeddings: Option<HashMap<String, Vec<f32>>>,
    pub query_templates: Vec<QueryTemplate>,
    pub statistics: DatasetStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTemplate {
    pub name: String,
    pub sparql: String,
    pub complexity: DatasetComplexity,
    pub expected_result_count: Option<usize>,
    pub estimated_execution_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub triple_count: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub avg_subject_degree: f64,
    pub avg_predicate_frequency: f64,
    pub graph_density: f64,
}

pub struct DatasetManager {
    datasets: HashMap<String, Dataset>,
    base_path: String,
}

impl DatasetManager {
    pub fn new(base_path: String) -> Self {
        Self {
            datasets: HashMap::new(),
            base_path,
        }
    }

    pub async fn generate_validation_datasets(&mut self) -> Result<()> {
        println!("📊 Generating validation datasets...");

        // Generate DBpedia-like dataset
        let dbpedia_config = DatasetConfig {
            name: "dbpedia_subset".to_string(),
            size: DatasetSize::Large,
            format: DatasetFormat::NTriples,
            complexity: DatasetComplexity::Complex,
            include_embeddings: true,
            embedding_dimensions: 512,
            domain: DatasetDomain::General,
        };
        let dbpedia_dataset = self.generate_dbpedia_dataset(dbpedia_config).await?;
        self.datasets.insert("dbpedia_subset".to_string(), dbpedia_dataset);

        // Generate Wikidata-like dataset
        let wikidata_config = DatasetConfig {
            name: "wikidata_subset".to_string(),
            size: DatasetSize::XLarge,
            format: DatasetFormat::Turtle,
            complexity: DatasetComplexity::VeryComplex,
            include_embeddings: true,
            embedding_dimensions: 768,
            domain: DatasetDomain::Cultural,
        };
        let wikidata_dataset = self.generate_wikidata_dataset(wikidata_config).await?;
        self.datasets.insert("wikidata_subset".to_string(), wikidata_dataset);

        // Generate biomedical dataset
        let biomed_config = DatasetConfig {
            name: "biomedical_ontology".to_string(),
            size: DatasetSize::Medium,
            format: DatasetFormat::RdfXml,
            complexity: DatasetComplexity::Complex,
            include_embeddings: true,
            embedding_dimensions: 256,
            domain: DatasetDomain::Biomedical,
        };
        let biomed_dataset = self.generate_biomedical_dataset(biomed_config).await?;
        self.datasets.insert("biomedical_ontology".to_string(), biomed_dataset);

        // Generate synthetic performance test dataset
        let synthetic_config = DatasetConfig {
            name: "synthetic_performance".to_string(),
            size: DatasetSize::XXLarge,
            format: DatasetFormat::NTriples,
            complexity: DatasetComplexity::Medium,
            include_embeddings: true,
            embedding_dimensions: 1024,
            domain: DatasetDomain::Technology,
        };
        let synthetic_dataset = self.generate_synthetic_dataset(synthetic_config).await?;
        self.datasets.insert("synthetic_performance".to_string(), synthetic_dataset);

        println!("✅ Generated {} validation datasets", self.datasets.len());
        Ok(())
    }

    async fn generate_dbpedia_dataset(&self, config: DatasetConfig) -> Result<Dataset> {
        println!("🌐 Generating DBpedia-like dataset...");

        let triple_count = self.get_triple_count_for_size(&config.size);
        let mut triples = Vec::with_capacity(triple_count);
        let mut rng = Random::seed(42);
        let uniform = Uniform::new(0, 1000);

        // Generate DBpedia-style entities and predicates
        let entities = self.generate_dbpedia_entities(&mut rng, triple_count / 10);
        let predicates = self.generate_dbpedia_predicates();

        // Generate triples with realistic DBpedia patterns
        for i in 0..triple_count {
            let subject = format!("dbr:{}", entities[rng.random_range(0..entities.len())]);
            let predicate = format!("dbo:{}", predicates[rng.random_range(0..predicates.len())]);

            let object = if rng.random_bool() {
                // Object property (another entity)
                format!("dbr:{}", entities[rng.random_range(0..entities.len())])
            } else {
                // Data property (literal)
                self.generate_dbpedia_literal(&mut rng)
            };

            triples.push(Triple {
                subject,
                predicate,
                object,
            });

            if i % 10000 == 0 {
                println!("Generated {} triples...", i);
            }
        }

        // Generate embeddings if requested
        let embeddings = if config.include_embeddings {
            Some(self.generate_embeddings_for_triples(&triples, config.embedding_dimensions, &mut rng))
        } else {
            None
        };

        // Generate SPARQL query templates
        let query_templates = self.generate_dbpedia_query_templates();

        // Calculate statistics
        let statistics = self.calculate_dataset_statistics(&triples);

        println!("✅ Generated DBpedia dataset with {} triples", triples.len());

        Ok(Dataset {
            config,
            triples,
            embeddings,
            query_templates,
            statistics,
        })
    }

    async fn generate_wikidata_dataset(&self, config: DatasetConfig) -> Result<Dataset> {
        println!("📚 Generating Wikidata-like dataset...");

        let triple_count = self.get_triple_count_for_size(&config.size);
        let mut triples = Vec::with_capacity(triple_count);
        let mut rng = Random::seed(123);
        let uniform = Uniform::new(0, 1000);

        // Generate Wikidata-style Q-entities and P-properties
        let q_entities = (1..=triple_count/5)
            .map(|i| format!("Q{}", i))
            .collect::<Vec<_>>();
        let p_properties = self.generate_wikidata_properties();

        for i in 0..triple_count {
            let subject = format!("wd:{}", q_entities[rng.random_range(0..q_entities.len())]);
            let predicate = format!("wdt:{}", p_properties[rng.random_range(0..p_properties.len())]);

            let object = if rng.random_bool() {
                format!("wd:{}", q_entities[rng.random_range(0..q_entities.len())])
            } else {
                self.generate_wikidata_literal(&mut rng)
            };

            triples.push(Triple {
                subject,
                predicate,
                object,
            });

            if i % 10000 == 0 {
                println!("Generated {} triples...", i);
            }
        }

        let embeddings = if config.include_embeddings {
            Some(self.generate_embeddings_for_triples(&triples, config.embedding_dimensions, &mut rng))
        } else {
            None
        };

        let query_templates = self.generate_wikidata_query_templates();
        let statistics = self.calculate_dataset_statistics(&triples);

        println!("✅ Generated Wikidata dataset with {} triples", triples.len());

        Ok(Dataset {
            config,
            triples,
            embeddings,
            query_templates,
            statistics,
        })
    }

    async fn generate_biomedical_dataset(&self, config: DatasetConfig) -> Result<Dataset> {
        println!("🧬 Generating biomedical dataset...");

        let triple_count = self.get_triple_count_for_size(&config.size);
        let mut triples = Vec::with_capacity(triple_count);
        let mut rng = Random::seed(456);
        let uniform = Uniform::new(0, 1000);

        // Generate biomedical entities and predicates
        let diseases = self.generate_disease_entities(&mut rng, 1000);
        let drugs = self.generate_drug_entities(&mut rng, 2000);
        let genes = self.generate_gene_entities(&mut rng, 5000);
        let proteins = self.generate_protein_entities(&mut rng, 3000);
        let predicates = self.generate_biomedical_predicates();

        let all_entities = [&diseases[..], &drugs[..], &genes[..], &proteins[..]].concat();

        for i in 0..triple_count {
            let subject = format!("bio:{}", all_entities[rng.random_range(0..all_entities.len())]);
            let predicate = format!("bioprop:{}", predicates[rng.random_range(0..predicates.len())]);
            let object = format!("bio:{}", all_entities[rng.random_range(0..all_entities.len())]);

            triples.push(Triple {
                subject,
                predicate,
                object,
            });

            if i % 5000 == 0 {
                println!("Generated {} triples...", i);
            }
        }

        let embeddings = if config.include_embeddings {
            Some(self.generate_embeddings_for_triples(&triples, config.embedding_dimensions, &mut rng))
        } else {
            None
        };

        let query_templates = self.generate_biomedical_query_templates();
        let statistics = self.calculate_dataset_statistics(&triples);

        println!("✅ Generated biomedical dataset with {} triples", triples.len());

        Ok(Dataset {
            config,
            triples,
            embeddings,
            query_templates,
            statistics,
        })
    }

    async fn generate_synthetic_dataset(&self, config: DatasetConfig) -> Result<Dataset> {
        println!("⚙️  Generating synthetic performance dataset...");

        let triple_count = self.get_triple_count_for_size(&config.size);
        let mut triples = Vec::with_capacity(triple_count);
        let mut rng = Random::seed(789);
        let uniform = Uniform::new(0, triple_count);

        // Generate synthetic entities and predicates for performance testing
        let subjects = (0..triple_count/100)
            .map(|i| format!("subj_{}", i))
            .collect::<Vec<_>>();
        let predicates = (0..100)
            .map(|i| format!("pred_{}", i))
            .collect::<Vec<_>>();
        let objects = (0..triple_count/50)
            .map(|i| format!("obj_{}", i))
            .collect::<Vec<_>>();

        for i in 0..triple_count {
            let subject = format!("synth:{}", subjects[rng.random_range(0..subjects.len())]);
            let predicate = format!("synthprop:{}", predicates[rng.random_range(0..predicates.len())]);
            let object = format!("synth:{}", objects[rng.random_range(0..objects.len())]);

            triples.push(Triple {
                subject,
                predicate,
                object,
            });

            if i % 100000 == 0 {
                println!("Generated {} triples...", i);
            }
        }

        let embeddings = if config.include_embeddings {
            Some(self.generate_embeddings_for_triples(&triples, config.embedding_dimensions, &mut rng))
        } else {
            None
        };

        let query_templates = self.generate_synthetic_query_templates();
        let statistics = self.calculate_dataset_statistics(&triples);

        println!("✅ Generated synthetic dataset with {} triples", triples.len());

        Ok(Dataset {
            config,
            triples,
            embeddings,
            query_templates,
            statistics,
        })
    }

    fn get_triple_count_for_size(&self, size: &DatasetSize) -> usize {
        match size {
            DatasetSize::Small => 1_000,
            DatasetSize::Medium => 10_000,
            DatasetSize::Large => 100_000,
            DatasetSize::XLarge => 1_000_000,
            DatasetSize::XXLarge => 10_000_000,
        }
    }

    fn generate_dbpedia_entities<R: rand::Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<String> {
        let entity_types = vec![
            "Person", "Place", "Organization", "Work", "Species", "Event",
            "Device", "Food", "ChemicalSubstance", "Disease", "Book", "Film",
        ];

        let mut entities = Vec::with_capacity(count);
        for i in 0..count {
            let entity_type = &entity_types[rng.random_range(0..entity_types.len())];
            entities.push(format!("{}_{}", entity_type, i));
        }
        entities
    }

    fn generate_dbpedia_predicates(&self) -> Vec<String> {
        vec![
            "type".to_string(),
            "label".to_string(),
            "comment".to_string(),
            "birthDate".to_string(),
            "deathDate".to_string(),
            "birthPlace".to_string(),
            "occupation".to_string(),
            "nationality".to_string(),
            "location".to_string(),
            "foundingDate".to_string(),
            "dissolutionDate".to_string(),
            "genre".to_string(),
            "director".to_string(),
            "starring".to_string(),
            "author".to_string(),
            "publisher".to_string(),
            "populationTotal".to_string(),
            "area".to_string(),
            "elevation".to_string(),
            "timeZone".to_string(),
        ]
    }

    fn generate_dbpedia_literal<R: rand::Rng>(&self, rng: &mut Random<R>) -> String {
        let literal_types = vec![
            format!("\"{}\"^^xsd:string", self.generate_random_string(rng, 20)),
            format!("\"{}\"^^xsd:int", rng.random_range(1..10000)),
            format!("\"{}\"^^xsd:date", self.generate_random_date(rng)),
            format!("\"{}\"@en", self.generate_random_text(rng, 50)),
        ];

        literal_types[rng.random_range(0..literal_types.len())].clone()
    }

    fn generate_wikidata_properties(&self) -> Vec<String> {
        vec![
            "P31".to_string(),  // instance of
            "P279".to_string(), // subclass of
            "P569".to_string(), // date of birth
            "P570".to_string(), // date of death
            "P19".to_string(),  // place of birth
            "P20".to_string(),  // place of death
            "P106".to_string(), // occupation
            "P27".to_string(),  // country of citizenship
            "P17".to_string(),  // country
            "P131".to_string(), // located in
            "P625".to_string(), // coordinate location
            "P577".to_string(), // publication date
            "P50".to_string(),  // author
            "P57".to_string(),  // director
            "P161".to_string(), // cast member
            "P136".to_string(), // genre
            "P175".to_string(), // performer
            "P1476".to_string(), // title
            "P18".to_string(),  // image
            "P154".to_string(), // logo image
        ]
    }

    fn generate_wikidata_literal<R: rand::Rng>(&self, rng: &mut Random<R>) -> String {
        let literal_types = vec![
            format!("\"{}\"", self.generate_random_string(rng, 30)),
            format!("{}", rng.random_range(1..1000000)),
            format!("\"{}-{:02}-{:02}T00:00:00Z\"",
                     rng.random_range(1900..2024),
                     rng.random_range(1..13),
                     rng.random_range(1..29)),
            format!("+{}.{}/{}",
                     rng.random_range(-90, 91),
                     rng.random_range(-180, 181),
                     "WGS84"),
        ];

        literal_types[rng.random_range(0..literal_types.len())].clone()
    }

    fn generate_disease_entities<R: rand::Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<String> {
        let disease_prefixes = vec![
            "cancer", "diabetes", "hypertension", "infection", "syndrome",
            "disorder", "disease", "condition", "inflammation", "deficiency",
        ];

        (0..count)
            .map(|i| {
                let prefix = &disease_prefixes[rng.random_range(0..disease_prefixes.len())];
                format!("{}_{}", prefix, i)
            })
            .collect()
    }

    fn generate_drug_entities<R: rand::Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<String> {
        let drug_suffixes = vec![
            "ine", "ol", "ide", "ate", "ose", "cin", "mycin", "pril", "sartan", "statin",
        ];

        (0..count)
            .map(|i| {
                let suffix = &drug_suffixes[rng.random_range(0..drug_suffixes.len())];
                format!("drug{}_{}", i, suffix)
            })
            .collect()
    }

    fn generate_gene_entities<R: rand::Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<String> {
        let gene_prefixes = vec![
            "BRCA", "TP53", "EGFR", "KRAS", "PIK3CA", "APC", "PTEN", "RB1", "MYC", "ATM",
        ];

        (0..count)
            .map(|i| {
                let prefix = &gene_prefixes[rng.random_range(0..gene_prefixes.len())];
                format!("{}{}", prefix, i)
            })
            .collect()
    }

    fn generate_protein_entities<R: rand::Rng>(&self, rng: &mut Random<R>, count: usize) -> Vec<String> {
        let protein_types = vec![
            "kinase", "phosphatase", "receptor", "channel", "transporter",
            "enzyme", "antibody", "cytokine", "hormone", "factor",
        ];

        (0..count)
            .map(|i| {
                let ptype = &protein_types[rng.random_range(0..protein_types.len())];
                format!("{}_{}", ptype, i)
            })
            .collect()
    }

    fn generate_biomedical_predicates(&self) -> Vec<String> {
        vec![
            "treats".to_string(),
            "causedBy".to_string(),
            "associatedWith".to_string(),
            "regulatedBy".to_string(),
            "interactsWith".to_string(),
            "expressedIn".to_string(),
            "locatedIn".to_string(),
            "participatesIn".to_string(),
            "hasFunction".to_string(),
            "hasStructure".to_string(),
            "metabolizedBy".to_string(),
            "transportedBy".to_string(),
            "synthesizedBy".to_string(),
            "degradedBy".to_string(),
            "activates".to_string(),
            "inhibits".to_string(),
            "bindsTo".to_string(),
            "encodedBy".to_string(),
            "translatedTo".to_string(),
            "transcribedFrom".to_string(),
        ]
    }

    fn generate_embeddings_for_triples<R: rand::Rng>(
        &self,
        triples: &[Triple],
        dimensions: usize,
        rng: &mut Random<R>,
    ) -> HashMap<String, Vec<f32>> {
        let mut embeddings = HashMap::new();

        // Extract all unique entities
        let mut entities = HashSet::new();
        for triple in triples {
            entities.insert(&triple.subject);
            entities.insert(&triple.predicate);
            entities.insert(&triple.object);
        }

        println!("Generating embeddings for {} unique entities...", entities.len());

        for (i, entity) in entities.iter().enumerate() {
            let embedding: Vec<f32> = (0..dimensions)
                .map(|_| generate_normal(rng, 0.0, 1.0) as f32)
                .collect();
            embeddings.insert((*entity).clone(), embedding);

            if i % 10000 == 0 {
                println!("Generated embeddings for {} entities...", i);
            }
        }

        embeddings
    }

    fn generate_dbpedia_query_templates(&self) -> Vec<QueryTemplate> {
        vec![
            QueryTemplate {
                name: "simple_entity_lookup".to_string(),
                sparql: r#"
                    PREFIX dbo: <http://dbpedia.org/ontology/>
                    PREFIX dbr: <http://dbpedia.org/resource/>
                    SELECT ?subject ?predicate ?object WHERE {
                        ?subject ?predicate ?object .
                        FILTER(?subject = dbr:Person_1)
                    } LIMIT 100
                "#.to_string(),
                complexity: DatasetComplexity::Simple,
                expected_result_count: Some(100),
                estimated_execution_time_ms: Some(50),
            },
            QueryTemplate {
                name: "complex_person_query".to_string(),
                sparql: r#"
                    PREFIX dbo: <http://dbpedia.org/ontology/>
                    PREFIX dbr: <http://dbpedia.org/resource/>
                    SELECT ?person ?birthPlace ?occupation WHERE {
                        ?person a dbo:Person .
                        ?person dbo:birthPlace ?birthPlace .
                        ?person dbo:occupation ?occupation .
                        ?birthPlace dbo:location ?country .
                        FILTER(?country = dbr:Place_1)
                    } LIMIT 1000
                "#.to_string(),
                complexity: DatasetComplexity::Complex,
                expected_result_count: Some(1000),
                estimated_execution_time_ms: Some(500),
            },
            QueryTemplate {
                name: "aggregation_query".to_string(),
                sparql: r#"
                    PREFIX dbo: <http://dbpedia.org/ontology/>
                    SELECT ?occupation (COUNT(?person) as ?count) WHERE {
                        ?person a dbo:Person .
                        ?person dbo:occupation ?occupation .
                    } GROUP BY ?occupation
                    ORDER BY DESC(?count)
                    LIMIT 50
                "#.to_string(),
                complexity: DatasetComplexity::Medium,
                expected_result_count: Some(50),
                estimated_execution_time_ms: Some(200),
            },
        ]
    }

    fn generate_wikidata_query_templates(&self) -> Vec<QueryTemplate> {
        vec![
            QueryTemplate {
                name: "wikidata_basic_lookup".to_string(),
                sparql: r#"
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    SELECT ?item ?property ?value WHERE {
                        ?item ?property ?value .
                        FILTER(?item = wd:Q1)
                    } LIMIT 100
                "#.to_string(),
                complexity: DatasetComplexity::Simple,
                expected_result_count: Some(100),
                estimated_execution_time_ms: Some(75),
            },
            QueryTemplate {
                name: "wikidata_complex_join".to_string(),
                sparql: r#"
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    SELECT ?person ?birthPlace ?country WHERE {
                        ?person wdt:P31 wd:Q5 .
                        ?person wdt:P19 ?birthPlace .
                        ?birthPlace wdt:P17 ?country .
                        ?person wdt:P569 ?birthDate .
                        FILTER(YEAR(?birthDate) > 1950)
                    } LIMIT 5000
                "#.to_string(),
                complexity: DatasetComplexity::VeryComplex,
                expected_result_count: Some(5000),
                estimated_execution_time_ms: Some(2000),
            },
        ]
    }

    fn generate_biomedical_query_templates(&self) -> Vec<QueryTemplate> {
        vec![
            QueryTemplate {
                name: "drug_disease_interaction".to_string(),
                sparql: r#"
                    PREFIX bio: <http://biomedical.org/entity/>
                    PREFIX bioprop: <http://biomedical.org/property/>
                    SELECT ?drug ?disease WHERE {
                        ?drug bioprop:treats ?disease .
                        ?disease bioprop:causedBy ?gene .
                        ?gene bioprop:associatedWith ?protein .
                    } LIMIT 1000
                "#.to_string(),
                complexity: DatasetComplexity::Complex,
                expected_result_count: Some(1000),
                estimated_execution_time_ms: Some(300),
            },
            QueryTemplate {
                name: "gene_protein_pathway".to_string(),
                sparql: r#"
                    PREFIX bio: <http://biomedical.org/entity/>
                    PREFIX bioprop: <http://biomedical.org/property/>
                    SELECT ?gene ?protein ?pathway WHERE {
                        ?gene bioprop:encodedBy ?protein .
                        ?protein bioprop:participatesIn ?pathway .
                        ?pathway bioprop:regulatedBy ?regulatory_gene .
                        ?regulatory_gene bioprop:interactsWith ?gene .
                    } LIMIT 2000
                "#.to_string(),
                complexity: DatasetComplexity::VeryComplex,
                expected_result_count: Some(2000),
                estimated_execution_time_ms: Some(1500),
            },
        ]
    }

    fn generate_synthetic_query_templates(&self) -> Vec<QueryTemplate> {
        vec![
            QueryTemplate {
                name: "synthetic_basic_pattern".to_string(),
                sparql: r#"
                    PREFIX synth: <http://synthetic.org/entity/>
                    PREFIX synthprop: <http://synthetic.org/property/>
                    SELECT ?s ?p ?o WHERE {
                        ?s ?p ?o .
                    } LIMIT 10000
                "#.to_string(),
                complexity: DatasetComplexity::Simple,
                expected_result_count: Some(10000),
                estimated_execution_time_ms: Some(100),
            },
            QueryTemplate {
                name: "synthetic_performance_test".to_string(),
                sparql: r#"
                    PREFIX synth: <http://synthetic.org/entity/>
                    PREFIX synthprop: <http://synthetic.org/property/>
                    SELECT ?subj (COUNT(?obj) as ?count) WHERE {
                        ?subj synthprop:pred_1 ?obj .
                        ?subj synthprop:pred_2 ?obj2 .
                        ?obj synthprop:pred_3 ?obj3 .
                    } GROUP BY ?subj
                    ORDER BY DESC(?count)
                    LIMIT 100000
                "#.to_string(),
                complexity: DatasetComplexity::Medium,
                expected_result_count: Some(100000),
                estimated_execution_time_ms: Some(5000),
            },
        ]
    }

    fn calculate_dataset_statistics(&self, triples: &[Triple]) -> DatasetStatistics {
        let mut subjects = HashSet::new();
        let mut predicates = HashSet::new();
        let mut objects = HashSet::new();
        let mut subject_degrees = HashMap::new();
        let mut predicate_counts = HashMap::new();

        for triple in triples {
            subjects.insert(&triple.subject);
            predicates.insert(&triple.predicate);
            objects.insert(&triple.object);

            *subject_degrees.entry(&triple.subject).or_insert(0) += 1;
            *predicate_counts.entry(&triple.predicate).or_insert(0) += 1;
        }

        let avg_subject_degree = if subjects.is_empty() {
            0.0
        } else {
            subject_degrees.values().sum::<usize>() as f64 / subjects.len() as f64
        };

        let avg_predicate_frequency = if predicates.is_empty() {
            0.0
        } else {
            predicate_counts.values().sum::<usize>() as f64 / predicates.len() as f64
        };

        let total_possible_edges = subjects.len() * objects.len();
        let graph_density = if total_possible_edges == 0 {
            0.0
        } else {
            triples.len() as f64 / total_possible_edges as f64
        };

        DatasetStatistics {
            triple_count: triples.len(),
            unique_subjects: subjects.len(),
            unique_predicates: predicates.len(),
            unique_objects: objects.len(),
            avg_subject_degree,
            avg_predicate_frequency,
            graph_density,
        }
    }

    fn generate_random_string<R: rand::Rng>(&self, rng: &mut Random<R>, max_len: usize) -> String {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let len = rng.random_range(5..max_len + 1);
        (0..len)
            .map(|_| {
                let idx = rng.random_range(0..chars.len());
                chars.chars().nth(idx).unwrap()
            })
            .collect()
    }

    fn generate_random_text<R: rand::Rng>(&self, rng: &mut Random<R>, max_len: usize) -> String {
        let words = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "about", "through", "during", "before", "after", "above", "below",
            "person", "place", "thing", "idea", "concept", "system", "process", "method",
            "way", "time", "year", "day", "week", "month", "hour", "minute", "second",
        ];

        let word_count = rng.random_range(3..12);
        let text_words: Vec<&str> = (0..word_count)
            .map(|_| words[rng.random_range(0..words.len())])
            .collect();

        text_words.join(" ")
    }

    fn generate_random_date<R: rand::Rng>(&self, rng: &mut Random<R>) -> String {
        let year = rng.random_range(1900..2024);
        let month = rng.random_range(1..13);
        let day = rng.random_range(1..29);
        format!("{}-{:02}-{:02}", year, month, day)
    }

    pub fn get_dataset(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }

    pub fn list_datasets(&self) -> Vec<&str> {
        self.datasets.keys().map(|s| s.as_str()).collect()
    }

    pub async fn save_datasets_to_disk(&self) -> Result<()> {
        for (name, dataset) in &self.datasets {
            let dataset_path = format!("{}/{}", self.base_path, name);
            fs::create_dir_all(&dataset_path)?;

            // Save triples
            let triples_path = format!("{}/triples.nt", dataset_path);
            self.save_triples_to_file(&dataset.triples, &triples_path, &dataset.config.format)?;

            // Save embeddings if present
            if let Some(embeddings) = &dataset.embeddings {
                let embeddings_path = format!("{}/embeddings.json", dataset_path);
                let embeddings_json = serde_json::to_string_pretty(embeddings)?;
                fs::write(embeddings_path, embeddings_json)?;
            }

            // Save query templates
            let queries_path = format!("{}/queries.sparql", dataset_path);
            self.save_query_templates(&dataset.query_templates, &queries_path)?;

            // Save metadata
            let metadata_path = format!("{}/metadata.json", dataset_path);
            let metadata = serde_json::to_string_pretty(&dataset.config)?;
            fs::write(metadata_path, metadata)?;

            println!("💾 Saved dataset '{}' to {}", name, dataset_path);
        }

        Ok(())
    }

    fn save_triples_to_file(&self, triples: &[Triple], path: &str, format: &DatasetFormat) -> Result<()> {
        let mut content = String::new();

        match format {
            DatasetFormat::NTriples => {
                for triple in triples {
                    content.push_str(&format!("<{}> <{}> <{}> .\n",
                                              triple.subject, triple.predicate, triple.object));
                }
            }
            DatasetFormat::Turtle => {
                content.push_str("@prefix dbr: <http://dbpedia.org/resource/> .\n");
                content.push_str("@prefix dbo: <http://dbpedia.org/ontology/> .\n");
                content.push_str("@prefix wd: <http://www.wikidata.org/entity/> .\n");
                content.push_str("@prefix wdt: <http://www.wikidata.org/prop/direct/> .\n\n");

                for triple in triples {
                    content.push_str(&format!("{} {} {} .\n",
                                              triple.subject, triple.predicate, triple.object));
                }
            }
            DatasetFormat::RdfXml | DatasetFormat::JsonLd => {
                // For now, save as N-Triples
                for triple in triples {
                    content.push_str(&format!("<{}> <{}> <{}> .\n",
                                              triple.subject, triple.predicate, triple.object));
                }
            }
        }

        fs::write(path, content)?;
        Ok(())
    }

    fn save_query_templates(&self, templates: &[QueryTemplate], path: &str) -> Result<()> {
        let mut content = String::new();

        for template in templates {
            content.push_str(&format!("# Query: {}\n", template.name));
            content.push_str(&format!("# Complexity: {:?}\n", template.complexity));
            if let Some(count) = template.expected_result_count {
                content.push_str(&format!("# Expected results: {}\n", count));
            }
            if let Some(time) = template.estimated_execution_time_ms {
                content.push_str(&format!("# Estimated time: {}ms\n", time));
            }
            content.push_str(&template.sparql);
            content.push_str("\n\n");
        }

        fs::write(path, content)?;
        Ok(())
    }
}