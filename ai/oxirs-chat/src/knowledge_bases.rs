//! External Knowledge Base Integrations
//!
//! Provides connectors for Wikipedia, PubMed, and other external knowledge sources
//! to enhance RAG capabilities with authoritative external data.

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Knowledge base query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseResult {
    /// Source knowledge base
    pub source: String,
    /// Result title
    pub title: String,
    /// Content/summary
    pub content: String,
    /// Full URL to the resource
    pub url: String,
    /// Relevance score (0.0 - 1.0)
    pub relevance: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Wikipedia API connector
pub struct WikipediaConnector {
    client: Client,
    api_endpoint: String,
    language: String,
}

impl WikipediaConnector {
    /// Create a new Wikipedia connector
    pub fn new(language: Option<String>) -> Result<Self> {
        let language = language.unwrap_or_else(|| "en".to_string());
        let api_endpoint = format!("https://{}.wikipedia.org/w/api.php", language);

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("OxiRS-Chat/0.1.0")
            .build()
            .context("Failed to create HTTP client for Wikipedia")?;

        info!("Initialized Wikipedia connector for language: {}", language);

        Ok(Self {
            client,
            api_endpoint,
            language,
        })
    }

    /// Search Wikipedia articles
    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<KnowledgeBaseResult>> {
        debug!("Searching Wikipedia for: {}", query);

        // Use Wikipedia's opensearch API
        let response = self
            .client
            .get(&self.api_endpoint)
            .query(&[
                ("action", "opensearch"),
                ("search", query),
                ("limit", &max_results.to_string()),
                ("format", "json"),
            ])
            .send()
            .await
            .context("Failed to send request to Wikipedia API")?;

        if !response.status().is_success() {
            return Err(anyhow!("Wikipedia API error: {}", response.status()));
        }

        let search_results: WikipediaSearchResponse = response
            .json()
            .await
            .context("Failed to parse Wikipedia search response")?;

        let mut results = Vec::new();

        for i in 0..search_results.1.len() {
            let title = &search_results.1[i];
            let description = search_results.2.get(i).cloned().unwrap_or_default();
            let url = search_results.3.get(i).cloned().unwrap_or_default();

            // Get article content
            let content = self.get_article_summary(title).await.unwrap_or(description);

            results.push(KnowledgeBaseResult {
                source: "wikipedia".to_string(),
                title: title.clone(),
                content,
                url,
                relevance: 1.0 - (i as f32 / max_results as f32), // Decreasing relevance
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("language".to_string(), self.language.clone());
                    meta
                },
            });
        }

        info!(
            "Found {} Wikipedia results for query: {}",
            results.len(),
            query
        );
        Ok(results)
    }

    /// Get article summary
    async fn get_article_summary(&self, title: &str) -> Result<String> {
        let response = self
            .client
            .get(&self.api_endpoint)
            .query(&[
                ("action", "query"),
                ("prop", "extracts"),
                ("exintro", "1"),
                ("explaintext", "1"),
                ("titles", title),
                ("format", "json"),
            ])
            .send()
            .await?;

        let result: serde_json::Value = response.json().await?;

        // Extract summary from nested JSON structure
        if let Some(pages) = result["query"]["pages"].as_object() {
            for (_page_id, page_data) in pages {
                if let Some(extract) = page_data["extract"].as_str() {
                    return Ok(extract.to_string());
                }
            }
        }

        Err(anyhow!("Failed to extract summary from Wikipedia response"))
    }
}

/// PubMed API connector
pub struct PubMedConnector {
    client: Client,
    api_endpoint: String,
    api_key: Option<String>,
}

impl PubMedConnector {
    /// Create a new PubMed connector
    pub fn new(api_key: Option<String>) -> Result<Self> {
        let api_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils".to_string();

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("OxiRS-Chat/0.1.0 (contact: support@oxirs.org)")
            .build()
            .context("Failed to create HTTP client for PubMed")?;

        info!("Initialized PubMed connector");

        Ok(Self {
            client,
            api_endpoint,
            api_key,
        })
    }

    /// Search PubMed articles
    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<KnowledgeBaseResult>> {
        debug!("Searching PubMed for: {}", query);

        // Step 1: Search for PMIDs
        let pmids = self.search_pmids(query, max_results).await?;

        // Step 2: Fetch article details for each PMID
        let mut results = Vec::new();

        for (i, pmid) in pmids.iter().enumerate() {
            match self.fetch_article_details(pmid).await {
                Ok(article) => {
                    results.push(KnowledgeBaseResult {
                        source: "pubmed".to_string(),
                        title: article.title,
                        content: article.abstract_text,
                        url: format!("https://pubmed.ncbi.nlm.nih.gov/{}/", pmid),
                        relevance: 1.0 - (i as f32 / max_results as f32),
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("pmid".to_string(), pmid.clone());
                            meta.insert("authors".to_string(), article.authors.join(", "));
                            if let Some(doi) = article.doi {
                                meta.insert("doi".to_string(), doi);
                            }
                            meta
                        },
                    });
                }
                Err(e) => {
                    warn!("Failed to fetch PubMed article {}: {}", pmid, e);
                }
            }

            // Rate limiting: PubMed allows 3 requests/second without API key, 10 with key
            let delay_ms = if self.api_key.is_some() { 100 } else { 340 };
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }

        info!(
            "Found {} PubMed results for query: {}",
            results.len(),
            query
        );
        Ok(results)
    }

    /// Search for PMIDs matching the query
    async fn search_pmids(&self, query: &str, max_results: usize) -> Result<Vec<String>> {
        let max_results_str = max_results.to_string();
        let mut params = vec![
            ("db", "pubmed"),
            ("term", query),
            ("retmax", max_results_str.as_str()),
            ("retmode", "json"),
        ];

        if let Some(ref api_key) = self.api_key {
            params.push(("api_key", api_key));
        }

        let response = self
            .client
            .get(format!("{}/esearch.fcgi", self.api_endpoint))
            .query(&params)
            .send()
            .await
            .context("Failed to send search request to PubMed")?;

        if !response.status().is_success() {
            return Err(anyhow!("PubMed API error: {}", response.status()));
        }

        let search_result: PubMedSearchResponse = response
            .json()
            .await
            .context("Failed to parse PubMed search response")?;

        Ok(search_result.esearchresult.idlist)
    }

    /// Fetch article details for a specific PMID
    async fn fetch_article_details(&self, pmid: &str) -> Result<PubMedArticle> {
        let mut params = vec![("db", "pubmed"), ("id", pmid), ("retmode", "xml")];

        if let Some(ref api_key) = self.api_key {
            params.push(("api_key", api_key));
        }

        let response = self
            .client
            .get(format!("{}/efetch.fcgi", self.api_endpoint))
            .query(&params)
            .send()
            .await
            .context("Failed to fetch article from PubMed")?;

        let xml_text = response.text().await?;

        // Parse XML (simplified - in production, use a proper XML parser)
        self.parse_pubmed_xml(&xml_text)
    }

    /// Parse PubMed XML response (simplified implementation)
    fn parse_pubmed_xml(&self, xml: &str) -> Result<PubMedArticle> {
        // Extract title
        let title =
            Self::extract_xml_tag(xml, "ArticleTitle").unwrap_or_else(|| "Untitled".to_string());

        // Extract abstract
        let abstract_text = Self::extract_xml_tag(xml, "AbstractText")
            .unwrap_or_else(|| "No abstract available".to_string());

        // Extract authors (simplified)
        let authors = Self::extract_authors(xml);

        // Extract DOI
        let doi = Self::extract_doi(xml);

        Ok(PubMedArticle {
            title,
            abstract_text,
            authors,
            doi,
        })
    }

    fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
        let start_tag = format!("<{}>", tag);
        let end_tag = format!("</{}>", tag);

        let start_pos = xml.find(&start_tag)? + start_tag.len();
        let end_pos = xml[start_pos..].find(&end_tag)? + start_pos;

        Some(xml[start_pos..end_pos].trim().to_string())
    }

    fn extract_authors(xml: &str) -> Vec<String> {
        let mut authors = Vec::new();

        // Simple extraction - in production, use proper XML parser
        let mut current_pos = 0;
        while let Some(lastname_start) = xml[current_pos..].find("<LastName>") {
            let lastname_start = current_pos + lastname_start + 10;
            if let Some(lastname_end) = xml[lastname_start..].find("</LastName>") {
                let lastname = &xml[lastname_start..lastname_start + lastname_end];

                // Try to find ForeName
                let forename =
                    if let Some(forename_start) = xml[lastname_start..].find("<ForeName>") {
                        let forename_start = lastname_start + forename_start + 10;
                        xml[forename_start..]
                            .find("</ForeName>")
                            .map(|forename_end| &xml[forename_start..forename_start + forename_end])
                    } else {
                        None
                    };

                let full_name = if let Some(first) = forename {
                    format!("{} {}", first, lastname)
                } else {
                    lastname.to_string()
                };

                authors.push(full_name);
                current_pos = lastname_start + lastname_end;
            } else {
                break;
            }
        }

        authors
    }

    fn extract_doi(xml: &str) -> Option<String> {
        // Look for DOI in ArticleId elements
        if let Some(doi_start) = xml.find(r#"<ArticleId IdType="doi">"#) {
            let doi_start = doi_start + 25;
            if let Some(doi_end) = xml[doi_start..].find("</ArticleId>") {
                return Some(xml[doi_start..doi_start + doi_end].trim().to_string());
            }
        }
        None
    }
}

/// Knowledge base manager coordinating multiple sources
pub struct KnowledgeBaseManager {
    wikipedia: Option<WikipediaConnector>,
    pubmed: Option<PubMedConnector>,
}

impl KnowledgeBaseManager {
    /// Create a new knowledge base manager
    pub fn new() -> Self {
        Self {
            wikipedia: None,
            pubmed: None,
        }
    }

    /// Enable Wikipedia integration
    pub fn with_wikipedia(mut self, language: Option<String>) -> Result<Self> {
        self.wikipedia = Some(WikipediaConnector::new(language)?);
        Ok(self)
    }

    /// Enable PubMed integration
    pub fn with_pubmed(mut self, api_key: Option<String>) -> Result<Self> {
        self.pubmed = Some(PubMedConnector::new(api_key)?);
        Ok(self)
    }

    /// Search all enabled knowledge bases
    pub async fn search_all(
        &self,
        query: &str,
        max_results_per_source: usize,
    ) -> Result<Vec<KnowledgeBaseResult>> {
        let mut all_results = Vec::new();

        // Search Wikipedia
        if let Some(ref wikipedia) = self.wikipedia {
            match wikipedia.search(query, max_results_per_source).await {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("Wikipedia search failed: {}", e),
            }
        }

        // Search PubMed
        if let Some(ref pubmed) = self.pubmed {
            match pubmed.search(query, max_results_per_source).await {
                Ok(mut results) => all_results.append(&mut results),
                Err(e) => warn!("PubMed search failed: {}", e),
            }
        }

        // Sort by relevance
        all_results.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(all_results)
    }
}

impl Default for KnowledgeBaseManager {
    fn default() -> Self {
        Self::new()
    }
}

// Type definitions for API responses

type WikipediaSearchResponse = (String, Vec<String>, Vec<String>, Vec<String>);

#[derive(Debug, Deserialize)]
struct PubMedSearchResponse {
    esearchresult: PubMedSearchResult,
}

#[derive(Debug, Deserialize)]
struct PubMedSearchResult {
    idlist: Vec<String>,
}

#[derive(Debug)]
struct PubMedArticle {
    title: String,
    abstract_text: String,
    authors: Vec<String>,
    doi: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wikipedia_connector_creation() {
        let connector = WikipediaConnector::new(Some("en".to_string()));
        assert!(connector.is_ok());
    }

    #[tokio::test]
    async fn test_pubmed_connector_creation() {
        let connector = PubMedConnector::new(None);
        assert!(connector.is_ok());
    }

    #[tokio::test]
    async fn test_knowledge_base_manager() {
        let manager = KnowledgeBaseManager::new()
            .with_wikipedia(Some("en".to_string()))
            .unwrap();

        assert!(manager.wikipedia.is_some());
    }

    #[test]
    fn test_xml_tag_extraction() {
        let xml = "<ArticleTitle>Test Title</ArticleTitle>";
        let result = PubMedConnector::extract_xml_tag(xml, "ArticleTitle");
        assert_eq!(result, Some("Test Title".to_string()));
    }
}
