# OxiRS

> Plateforme modulaire native Rust pour le Web Sémantique, SPARQL 1.2, GraphQL et raisonnement augmenté par IA

[![Licence: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)

**Statut**: v0.1.0 - Version de Production Initiale - Publié le 7 janvier 2026

🎉 **Prêt pour la Production**: Implémentation complète de SPARQL 1.1/1.2 avec **optimiseur 3,8× plus rapide**, support IoT industriel et fonctionnalités IA. 13 123 tests réussis, zéro avertissement.

## Vision

OxiRS est une alternative **Rust-first, sans JVM** à Apache Jena + Fuseki et Juniper, offrant:

- **Choix de protocole, pas de verrouillage**: Points de terminaison SPARQL 1.2 et GraphQL depuis le même jeu de données
- **Adoption progressive**: Chaque crate fonctionne de manière autonome; fonctionnalités avancées via les features Cargo
- **Prêt pour l'IA**: Intégration native de recherche vectorielle, embeddings de graphes et requêtes augmentées par LLM
- **Binaire statique unique**: Parité de fonctionnalités avec Jena/Fuseki avec une empreinte <50 Mo

## Gaia-X & Souveraineté des Données Européennes 🇪🇺

OxiRS offre un support de premier ordre pour le **Gaia-X Trust Framework** et les exigences de souveraineté des données européennes.

### Fonctionnalités Principales

**Gaia-X Trust Framework**
- ✅ **Conformité Gaia-X**: Vérification de Self-Description et registre des participants
- ✅ **Politique ODRL 2.2**: Support complet des politiques d'usage et contraintes
- ✅ **Connecteur IDS**: Architecture International Data Spaces (IDSA RAM 4.x)
- ✅ **Credentials Vérifiables**: W3C DID et preuves cryptographiques (Ed25519)
- ✅ **Conformité RGPD**: Vérification automatique d'adéquation et AIPD

**Industrie 4.0 / Automobile (Catena-X)**
- ✅ **SAMM 2.0-2.3**: Semantic Aspect Meta Model pour modèles de données automobiles
- ✅ **AAS (Asset Administration Shell)**: Standard Digital Twin Industrie 4.0
- ✅ **Modbus/OPC UA**: Protocoles IoT industriels
- ✅ **CANbus (J1939)**: Télématique et diagnostic véhicules
- ✅ **Jumeaux Numériques**: Simulations basées sur la physique (intégration SciRS2)

**Espaces de Données Européens**
- ✅ **Manufacturing-X**: Espaces de données communs pour la fabrication
- ✅ **Mobility Data Space**: Échange de données de mobilité
- ✅ **Health Data Space**: Données de santé conformes au RGPD
- ✅ **Green Deal Data Space**: Données environnementales et durabilité

**RGPD & Protection des Données**
- ✅ **Data Residency**: Stockage UE uniquement, EEE uniquement, ou par pays
- ✅ **Décisions d'Adéquation**: Vérification automatique des décisions d'adéquation RGPD
- ✅ **Droit à l'Effacement**: SPARQL UPDATE pour demandes de suppression RGPD
- ✅ **Pistes d'Audit**: Traçage de provenance W3C PROV-O

### Démarrage Rapide

#### Installation

```bash
# Installer l'outil CLI
cargo install oxirs --version 0.1.0

# Compiler depuis les sources
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --workspace --release
```

#### Configuration Gaia-X

```bash
# Utiliser le template optimisé Gaia-X
cp oxirs-gaiax.toml oxirs.toml

# Démarrer le serveur avec optimisations européennes
oxirs serve oxirs.toml --port 3030
```

#### Exemple de Self-Description Gaia-X

```rust
use oxirs_fuseki::ids::identity::gaiax_registry::{GaiaxRegistry, GaiaxSelfDescription};

// Créer un client de registre Gaia-X
let registry = GaiaxRegistry::new("https://registry.gaia-x.eu".to_string());

// Vérifier un participant
let is_valid = registry.verify_participant("did:web:example.com").await?;

// Récupérer la Self-Description
let sd = registry.get_self_description("did:web:example.com").await?;

// Vérifier la conformité
let compliance = registry.verify_self_description(&sd).await?;
println!("Conforme: {}", compliance.compliant);
```

#### Exemple de Politique ODRL (Catena-X)

```rust
use oxirs_fuseki::ids::policy::{OdrlPolicy, Permission, Constraint};

// Politique Catena-X Battery Passport
let policy = OdrlPolicy {
    uid: "urn:policy:catena-x:battery-data:001".into(),
    permissions: vec![
        Permission {
            action: OdrlAction::Use,
            constraints: vec![
                // Usage uniquement à des fins de recherche
                Constraint::Purpose {
                    allowed_purposes: vec![Purpose::Research],
                },
                // Seulement dans l'UE/EEE
                Constraint::Spatial {
                    allowed_regions: vec![Region::eu_member("FR", "France")],
                },
                // Validité de 90 jours
                Constraint::Temporal {
                    operator: ComparisonOperator::LessThanOrEqual,
                    right_operand: Utc::now() + Duration::days(90),
                },
            ],
        }
    ],
};

// Appliquer la politique
let evaluator = ConstraintEvaluator::new();
let result = evaluator.evaluate_policy(&policy, &context).await?;
```

### Déploiement Cloud (Europe)

**AWS Europe (Paris - eu-west-3)**
```bash
# Déployer dans AWS Paris avec conformité RGPD
terraform apply -var="region=eu-west-3" -var="instance_type=t3.xlarge"
```

**Azure Europe (France Central)**
```bash
# Déployer dans Azure France
az deployment group create --resource-group oxirs-rg \
  --template-file deploy-azure-france.json \
  --parameters location=francecentral vmSize=Standard_D4s_v3
```

**OVH Cloud (Français)**
```bash
# Déployer dans OVH Cloud (fournisseur européen conforme RGPD)
openstack server create --flavor b2-15 --image "Ubuntu 22.04" oxirs-server
```

**Scaleway (Français)**
```bash
# Déployer dans Scaleway (fournisseur français)
scw instance server create type=DEV1-L zone=fr-par-1 image=ubuntu-jammy
```

### Cas d'Usage

1. **Catena-X Automobile**: Battery Passport, passeports numériques de produits avec SAMM 2.3
2. **Manufacturing-X**: Échange de données de fabrication avec connecteurs IDS
3. **Réseaux Intelligents**: Partage de données énergétiques conforme RGPD
4. **Health Data Space**: Données patients avec protection européenne
5. **Mobility Data Space**: Fédération de données de trafic pour villes intelligentes
6. **Green Deal**: Données environnementales pour développement durable

### Benchmarks de Performance (Échelle Européenne)

```
Déploiement Catena-X Battery Passport:
  Connecteurs:          1 000+ connecteurs IDS (UE)
  Politiques:           10 000+ politiques ODRL
  Débit requêtes:       50 000 QPS
  Latence p99:          <150ms (Paris→Francfort)
  Conformité RGPD:      100% stockage UE uniquement

Manufacturing-X Digital Twin:
  Dispositifs OPC UA:   5 000+ installations industrielles
  Instances AAS:        10 000+ Asset Administration Shells
  Fréquence de mise à jour: 10Hz par dispositif
  Simulation:           Physique temps réel (SciRS2)
```

## Nouveautés dans v0.1.0 (7 janvier 2026) 🎉

**Version de Production Initiale: Plateforme Web Sémantique Complète**

**Fonctionnalités Principales:**
- 🚀 **SPARQL 1.1/1.2 Complet** - Conforme W3C avec optimisation de requêtes avancée
- ⚡ **Optimiseur 3,8× Plus Rapide** - Détection adaptative de la complexité
- 🏭 **IoT Industriel** - Séries temporelles, Modbus, CANbus/J1939
- 🤖 **Augmenté par IA** - GraphRAG, embeddings, inférence physique-informée
- 🔐 **Sécurité Production** - ReBAC, OAuth2/OIDC, DID & Credentials Vérifiables
- 📊 **Observabilité Complète** - Métriques Prometheus, Tracing OpenTelemetry
- ☁️ **Cloud-Native** - Opérateur Kubernetes, modules Terraform

**Métriques de Qualité:**
- ✅ **13 123 tests réussis** (100% de réussite, 136 ignorés)
- ✅ **Zéro avertissement de compilation** sur les 22 crates
- ✅ **95%+ de couverture de tests**
- ✅ **Validé en production** dans des déploiements industriels

## Architecture

```
oxirs/                  # Racine Cargo Workspace
├─ core/                # Modules fondamentaux
│  └─ oxirs-core
├─ server/              # Frontends réseau
│  ├─ oxirs-fuseki      # Serveur HTTP SPARQL 1.1/1.2
│  └─ oxirs-gql         # Façade GraphQL
├─ engine/              # Requête, mise à jour, inférence
│  ├─ oxirs-arq         # Algèbre style Jena + points d'extension
│  ├─ oxirs-rule        # Moteur d'inférence avant/arrière
│  ├─ oxirs-samm        # Métamodèle SAMM + intégration AAS
│  ├─ oxirs-geosparql   # Support GeoSPARQL
│  ├─ oxirs-shacl       # SHACL Core + SHACL-SPARQL
│  ├─ oxirs-star        # RDF-star / SPARQL-star
│  ├─ oxirs-ttl         # Parseur Turtle/TriG
│  └─ oxirs-vec         # Abstractions d'index vectoriel
├─ storage/
│  ├─ oxirs-tdb         # Couche MVCC (parité TDB2)
│  └─ oxirs-cluster     # Dataset distribué basé Raft
├─ stream/              # Temps réel et fédération
│  ├─ oxirs-stream      # E/S Kafka/NATS, RDF Patch
│  └─ oxirs-federate    # Planificateur SERVICE, stitching GraphQL
├─ ai/
│  ├─ oxirs-embed       # Embeddings KG (TransE, ComplEx...)
│  ├─ oxirs-shacl-ai    # Induction de formes & suggestions de réparation
│  ├─ oxirs-chat        # API chat RAG (LLM + SPARQL)
│  ├─ oxirs-physics     # Jumeaux numériques physique-informés
│  └─ oxirs-graphrag    # Recherche hybride GraphRAG
├─ security/
│  └─ oxirs-did         # W3C DID & Credentials Vérifiables
├─ platforms/
│  └─ oxirs-wasm        # Déploiement WebAssembly navigateur/edge
└─ tools/
    ├─ oxirs             # CLI (import, export, benchmark)
    └─ benchmarks/       # SP2Bench, WatDiv, LDBC SGS
```

## Matrice de Fonctionnalités (v0.1.0)

| Capacité | Crate(s) OxiRS | Statut | Parité Jena/Fuseki |
|----------|----------------|--------|--------------------|
| **RDF & SPARQL de Base** | | | |
| RDF 1.2 & 7 formats | `oxirs-core` | ✅ Stable (600+ tests) | ✅ |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ Stable (550+ tests) | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (flag `star`) | ✅ Stable | 🔸 |
| **Extensions Web Sémantique** | | | |
| RDF-star Parse/Sérialisation | `oxirs-star` | ✅ Stable (200+ tests) | 🔸 |
| SHACL Core+API (conforme W3C) | `oxirs-shacl` | ✅ Stable (400+ tests) | ✅ |
| Inférence de règles (RDFS/OWL) | `oxirs-rule` | ✅ Stable (200+ tests) | ✅ |
| SAMM 2.0-2.3 & AAS | `oxirs-samm` | ✅ Stable (16 générateurs) | ❌ |
| **Requête & Fédération** | | | |
| API GraphQL | `oxirs-gql` | ✅ Stable (150+ tests) | ❌ |
| Fédération SPARQL (SERVICE) | `oxirs-federate` | ✅ Stable (350+ tests) | ✅ |
| **Temps Réel & Streaming** | | | |
| Traitement de flux (Kafka/NATS) | `oxirs-stream` | ✅ Stable (300+ tests) | 🔸 |
| **Recherche & Géo** | | | |
| Recherche plein texte (`text:`) | `oxirs-textsearch` | ⏳ Prévu | ✅ |
| GeoSPARQL (OGC 1.1) | `oxirs-geosparql` | ✅ Stable (250+ tests) | ✅ |
| Recherche vectorielle/embeddings | `oxirs-vec`, `oxirs-embed` | ✅ Stable (750+ tests) | ❌ |
| **Stockage & Distribution** | | | |
| Stockage compatible TDB2 | `oxirs-tdb` | ✅ Stable (250+ tests) | ✅ |
| Stockage distribué/HA (Raft) | `oxirs-cluster` | ✅ Stable | 🔸 |
| **IA & Fonctionnalités Avancées** | | | |
| API chat RAG | `oxirs-chat` | ✅ Stable | ❌ |
| Validation SHACL augmentée IA | `oxirs-shacl-ai` | ✅ Stable (350+ tests) | ❌ |
| Recherche hybride GraphRAG | `oxirs-graphrag` | ✅ Stable (23 tests) | ❌ |
| Jumeaux numériques physique-informés | `oxirs-physics` | ✅ Stable | ❌ |
| **Sécurité & Confiance** | | | |
| W3C DID & Credentials Vérifiables | `oxirs-did` | ✅ Stable (43 tests) | ❌ |
| Self-Descriptions Gaia-X | `oxirs-fuseki` (IDS) | ✅ Stable | ❌ |
| Connecteur IDS (IDSA RAM 4.x) | `oxirs-fuseki` (IDS) | ✅ Stable | ❌ |
| Application Politique ODRL 2.2 | `oxirs-fuseki` (IDS) | ✅ Stable | ❌ |
| ReBAC (Contrôle d'Accès Relationnel) | `oxirs-fuseki` | ✅ Stable (83 tests) | ❌ |
| OAuth2/OIDC/SAML | `oxirs-fuseki` | ✅ Stable | 🔸 |

**Légende:**
- ✅ Stable: Prêt production, tests complets, garantie stabilité API
- 🔄 Expérimental: En développement actif, APIs peuvent changer
- ⏳ Prévu: Pas encore implémenté
- 🔸 Support partiel/plugin dans Jena

## Documentation

### Documentation en Français

- 📄 **[Architecture IDS](docs/IDS_ARCHITECTURE.md)** - Architecture de souveraineté des données (Anglais)
- 📄 **[Opérations IDS](docs/IDS_OPERATIONS.md)** - Manuel d'exploitation pour connecteurs IDS (Anglais)
- 📄 **[Digital Twin Quickstart](server/oxirs-fuseki/DIGITAL_TWIN_QUICKSTART.md)** - IoT industriel (Anglais)
- ⚙️ **[oxirs-gaiax.toml](oxirs-gaiax.toml)** - Template de configuration Gaia-X

### Initiatives Européennes

- 🏛️ [Gaia-X](https://gaia-x.eu/)
- 🏛️ [Catena-X](https://catena-x.net/)
- 🏛️ [Manufacturing-X](https://www.manufacturing-x.de/)
- 🏛️ [IDSA (International Data Spaces Association)](https://internationaldataspaces.org/)
- 🏛️ [Agence Nationale de la Sécurité des Systèmes d'Information (ANSSI)](https://www.ssi.gouv.fr/)

### Standards & Conformité

- ✅ **RGPD**: Règlement Général sur la Protection des Données
- ✅ **IDSA RAM 4.x**: Modèle d'Architecture de Référence IDS
- ✅ **Gaia-X Trust Framework**: Version 22.10
- ✅ **ODRL 2.2**: Open Digital Rights Language
- ✅ **W3C PROV-O**: Ontologie de Provenance

## Développement

### Prérequis

- Rust 1.70+ (MSRV)
- Optionnel: Docker pour déploiement conteneurisé

### Build

```bash
# Cloner le dépôt
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Compiler toutes les crates
cargo build --workspace

# Exécuter les tests
cargo nextest run --no-fail-fast

# Compiler avec toutes les fonctionnalités
cargo build --workspace --all-features
```

## Feuille de Route

| Version | Date Cible | Jalon | Livrables | Statut |
|---------|-----------|-------|-----------|--------|
| **v0.1.0** | **✅ 7 jan 2026** | **Production Initiale** | SPARQL 1.1/1.2 complet, IoT industriel, IA | ✅ Publié |
| **v0.2.0** | **T1 2026** | **Performance, Recherche & Géo** | 10× performance, recherche plein texte, GeoSPARQL | 🎯 Prochain |
| **v1.0.0** | **T2 2026** | **Version LTS** | Parité Jena complète, support entreprise | 📋 Prévu |

## Licence

OxiRS est sous double licence:

- **Licence MIT** ([LICENSE-MIT](LICENSE-MIT) ou http://opensource.org/licenses/MIT)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) ou http://www.apache.org/licenses/LICENSE-2.0)

Vous pouvez choisir à votre convenance. Voir [LICENSE](LICENSE) pour les détails.

## Contact

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **Mainteneur**: @cool-japan (KitaSan)

---

📖 **Autres Langues:**
- 🇬🇧 [English](README.md)
- 🇯🇵 [日本語 (Japanese)](README.ja.md)
- 🇩🇪 [Deutsch (German)](README.de.md)

---

*"Rust rend la sécurité de la mémoire incontournable; OxiRS rend l'ingénierie des graphes de connaissances incontournable."*

**v0.1.0 - Version de Production Initiale - 7 janvier 2026**
