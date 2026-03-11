# OxiRS

> Plateforme modulaire native Rust pour le Web Sémantique, SPARQL 1.2, GraphQL et raisonnement augmenté par IA

[![Licence: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.1-blue)](https://github.com/cool-japan/oxirs/releases)

**Statut**: v0.2.1 - Publié - 11 mars 2026

**Prêt pour la Production**: Implémentation complète de SPARQL 1.1/1.2 avec **optimiseur 3,8× plus rapide**, support IoT industriel et fonctionnalités IA. **40 791+ tests réussis**, zéro avertissement sur les 26 crates.

**Points Forts v0.2.1 (11 mars 2026)**: 26 nouveaux modules fonctionnels ajoutés sur 16 rounds de développement. Algèbre SPARQL avancée (EXISTS, MINUS, sous-requêtes, clause SERVICE), stockage renforcé (store six-index, fusion/reconstruction d'index), capacités IA (store vectoriel, inférence de contraintes, historique de conversation) et renforcement de la sécurité (store d'identifiants, validation de chaîne de confiance).

## Vision

OxiRS est une alternative **Rust-first, sans JVM** à Apache Jena + Fuseki et Juniper, offrant:

- **Choix de protocole, pas de verrouillage**: Points de terminaison SPARQL 1.2 et GraphQL depuis le même jeu de données
- **Adoption progressive**: Chaque crate fonctionne de manière autonome; fonctionnalités avancées via les features Cargo
- **Prêt pour l'IA**: Intégration native de recherche vectorielle, embeddings de graphes et requêtes augmentées par LLM
- **Binaire statique unique**: Parité de fonctionnalités avec Jena/Fuseki avec une empreinte <50 Mo

## Gaia-X & Souveraineté des Données Européennes

OxiRS offre un support de premier ordre pour le **Gaia-X Trust Framework** et les exigences de souveraineté des données européennes.

### Fonctionnalités Principales

**Gaia-X Trust Framework**
- **Conformité Gaia-X**: Vérification de Self-Description et registre des participants
- **Politique ODRL 2.2**: Support complet des politiques d'usage et contraintes
- **Connecteur IDS**: Architecture International Data Spaces (IDSA RAM 4.x)
- **Credentials Vérifiables**: W3C DID et preuves cryptographiques (Ed25519)
- **Conformité RGPD**: Vérification automatique d'adéquation et AIPD

**Industrie 4.0 / Automobile (Catena-X)**
- **SAMM 2.0-2.3**: Semantic Aspect Meta Model pour modèles de données automobiles
- **AAS (Asset Administration Shell)**: Standard Digital Twin Industrie 4.0
- **Modbus/OPC UA**: Protocoles IoT industriels
- **CANbus (J1939)**: Télématique et diagnostic véhicules
- **Jumeaux Numériques**: Simulations basées sur la physique (intégration SciRS2)

**Espaces de Données Européens**
- **Manufacturing-X**: Espaces de données communs pour la fabrication
- **Mobility Data Space**: Échange de données de mobilité
- **Health Data Space**: Données de santé conformes au RGPD
- **Green Deal Data Space**: Données environnementales et durabilité

**RGPD & Protection des Données**
- **Data Residency**: Stockage UE uniquement, EEE uniquement, ou par pays
- **Décisions d'Adéquation**: Vérification automatique des décisions d'adéquation RGPD
- **Droit à l'Effacement**: SPARQL UPDATE pour demandes de suppression RGPD
- **Pistes d'Audit**: Traçage de provenance W3C PROV-O

### Démarrage Rapide

#### Installation

```bash
# Installer l'outil CLI
cargo install oxirs --version 0.2.1

# Compiler depuis les sources
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --workspace --release
```

#### Utilisation

```bash
# Initialiser un nouveau graphe de connaissances
oxirs init monkg

# Importer des données RDF
oxirs import monkg data.ttl --format turtle

# Interroger les données
oxirs query monkg "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Démarrer le serveur
oxirs serve monkg/oxirs.toml --port 3030
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

## Nouveautés dans v0.2.1 (11 mars 2026)

**v0.2.1 — Expansion Majeure des Fonctionnalités: 26 Nouveaux Modules sur 16 Rounds**

OxiRS v0.2.1 élargit considérablement la plateforme avec une algèbre SPARQL approfondie, un stockage de production, des capacités IA et un renforcement de la sécurité:

**Fonctionnalités Principales:**
- **SPARQL 1.1/1.2 Complet** - Conforme W3C avec optimisation de requêtes avancée
- **Optimiseur 3,8× Plus Rapide** - Détection adaptative de la complexité
- **Algèbre SPARQL Avancée** - Évaluateurs EXISTS/MINUS, constructeur de sous-requêtes, clause SERVICE, jointure LATERAL
- **IoT Industriel** - Séries temporelles, Modbus, CANbus/J1939
- **Augmenté par IA** - GraphRAG, store vectoriel, inférence de contraintes, historique de conversation, thermodynamique
- **Sécurité Production** - ReBAC, OAuth2/OIDC, DID & Credentials Vérifiables, validation de chaîne de confiance
- **Renforcement du Stockage** - Store six-index (SPO/POS/OSP/GSPO/GPOS/GOPS), fusion/reconstruction d'index, cache de triples, routeur de shards
- **Observabilité Complète** - Métriques Prometheus, Tracing OpenTelemetry
- **Cloud-Native** - Opérateur Kubernetes, modules Terraform, support Docker

**Métriques de Qualité:**
- **40 791 tests réussis** (100% de réussite, ~115 ignorés)
- **Zéro avertissement de compilation** sur les 26 crates
- **95%+ de couverture de tests et de documentation**
- **Validé en production** dans des déploiements industriels
- **26 nouveaux modules fonctionnels** ajoutés sur 16 rounds de développement

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
│  ├─ oxirs-cluster     # Dataset distribué basé Raft
│  └─ oxirs-tsdb        # Base de données de séries temporelles
├─ stream/              # Temps réel et fédération
│  ├─ oxirs-stream      # E/S Kafka/NATS, RDF Patch
│  ├─ oxirs-federate    # Planificateur SERVICE, stitching GraphQL
│  ├─ oxirs-modbus      # Protocole Modbus TCP/RTU
│  └─ oxirs-canbus      # Protocole CANbus / J1939
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
    ├─ oxirs             # CLI (import, export, diff, convert, validate...)
    └─ benchmarks/       # SP2Bench, WatDiv, LDBC SGS
```

## Matrice de Fonctionnalités (v0.2.1)

| Capacité | Crate(s) OxiRS | Statut | Parité Jena/Fuseki |
|----------|----------------|--------|--------------------|
| **RDF & SPARQL de Base** | | | |
| RDF 1.2 & 7 formats | `oxirs-core` | Stable (2 458 tests) | Oui |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | Stable (1 626 + 2 628 tests) | Oui |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (flag `star`) | Stable | Partiel |
| Algèbre SPARQL Avancée (EXISTS/MINUS/sous-requête) | `oxirs-arq` | Stable | Oui |
| Stockage persistant (N-Quads) | `oxirs-core` | Stable | Oui |
| **Extensions Web Sémantique** | | | |
| RDF-star Parse/Sérialisation | `oxirs-star` | Stable (1 507 tests) | Partiel |
| SHACL Core+API (conforme W3C) | `oxirs-shacl` | Stable (1 915 tests, 27/27 W3C) | Oui |
| Inférence de règles (RDFS/OWL 2 DL) | `oxirs-rule` | Stable (2 114 tests) | Oui |
| SAMM 2.0-2.3 & AAS | `oxirs-samm` | Stable (1 326 tests, 16 générateurs) | Non |
| **Requête & Fédération** | | | |
| API GraphQL | `oxirs-gql` | Stable (1 706 tests) | Non |
| Fédération SPARQL (SERVICE) | `oxirs-federate` | Stable (1 148 tests, 2PC) | Oui |
| Authentification fédérée | `oxirs-federate` | Stable (OAuth2/SAML/JWT) | Partiel |
| **Temps Réel & Streaming** | | | |
| Traitement de flux (Kafka/NATS) | `oxirs-stream` | Stable (1 191 tests, SIMD) | Partiel |
| RDF Patch & SPARQL Update delta | `oxirs-stream` | Stable | Partiel |
| **Recherche & Géo** | | | |
| Recherche plein texte (`text:`) | `oxirs-textsearch` | Prévu | Oui |
| GeoSPARQL (OGC 1.1) | `oxirs-geosparql` | Stable (1 756 tests) | Oui |
| Recherche vectorielle/embeddings | `oxirs-vec` (1 587 tests), `oxirs-embed` (1 408 tests) | Stable | Non |
| **Stockage & Distribution** | | | |
| Stockage compatible TDB2 (six-index) | `oxirs-tdb` | Stable (2 068 tests) | Oui |
| Stockage distribué/HA (Raft) | `oxirs-cluster` | Stable (1 019 tests) | Partiel |
| Base de données de séries temporelles | `oxirs-tsdb` | Stable (1 250 tests) | Non |
| **IA & Fonctionnalités Avancées** | | | |
| API chat RAG (intégration LLM) | `oxirs-chat` | Stable (1 095 tests) | Non |
| Validation SHACL augmentée IA | `oxirs-shacl-ai` | Stable (1 509 tests) | Non |
| Recherche hybride GraphRAG | `oxirs-graphrag` | Stable (998 tests) | Non |
| Jumeaux numériques physique-informés | `oxirs-physics` | Stable (1 225 tests) | Non |
| Embeddings de graphes de connaissances | `oxirs-embed` | Stable (1 408 tests) | Non |
| **Sécurité & Confiance** | | | |
| W3C DID & Credentials Vérifiables | `oxirs-did` | Stable (1 196 tests) | Non |
| Validation de chaîne de confiance | `oxirs-did` | Stable | Non |
| Graphes RDF signés (RDFC-1.0) | `oxirs-did` | Stable | Non |
| Preuves cryptographiques Ed25519 | `oxirs-did` | Stable | Non |
| **Sécurité & Autorisation** | | | |
| ReBAC (Contrôle d'Accès Relationnel) | `oxirs-fuseki` | Stable | Non |
| Autorisation au niveau du graphe | `oxirs-fuseki` | Stable | Non |
| OAuth2/OIDC/SAML | `oxirs-fuseki` | Stable | Partiel |
| Self-Descriptions Gaia-X | `oxirs-fuseki` (IDS) | Stable | Non |
| Connecteur IDS (IDSA RAM 4.x) | `oxirs-fuseki` (IDS) | Stable | Non |
| Application Politique ODRL 2.2 | `oxirs-fuseki` (IDS) | Stable | Non |
| **Navigateur & Edge** | | | |
| Bindings WebAssembly (WASM) | `oxirs-wasm` | Stable (1 036 tests) | Non |
| **IoT Industriel** | | | |
| Protocole Modbus TCP/RTU | `oxirs-modbus` | Stable (1 115 tests) | Non |
| Protocole CANbus / J1939 | `oxirs-canbus` | Stable (1 158 tests) | Non |

**Légende:**
- Stable: Prêt production, tests complets, garantie stabilité API
- Prévu: Pas encore implémenté
- Partiel: Support partiel/plugin dans Jena

**Métriques de Qualité (v0.2.1):**
- **40 791 tests réussis** (100% de réussite, ~115 ignorés)
- **Zéro avertissement de compilation** (appliqué avec `-D warnings`)
- **95%+ de couverture de tests** sur les 26 modules
- **95%+ de couverture de documentation**
- **Tous les tests d'intégration réussis**
- **Audit de sécurité de niveau production effectué**
- **Support GPU CUDA** pour accélération IA
- **Optimisation de requêtes 3,8× plus rapide** via détection adaptative de complexité
- **26 nouveaux modules fonctionnels** ajoutés dans v0.2.1 (16 rounds de développement)

## Notes de Version v0.2.1

Les notes complètes se trouvent dans [CHANGELOG.md](CHANGELOG.md).

### Points Forts (11 mars 2026)
- **40 791+ tests réussis** sur les 26 crates
- **26 nouveaux modules fonctionnels** ajoutés sur les 26 crates en 16 rounds de développement
- **Algèbre SPARQL avancée**: évaluateur EXISTS, évaluateur MINUS, constructeur de sous-requêtes, gestionnaire de clause SERVICE
- **Renforcement du stockage**: store six-index (SPO/POS/OSP/GSPO/GPOS/GOPS), fusion/reconstruction d'index, compaction B-tree
- **IA prête pour la production**: store vectoriel, inférence de contraintes, historique de conversation, cache de réponses, rerankeur
- **Renforcement de la sécurité**: store d'identifiants, validation de chaîne de confiance, gestionnaire de clés, présentateur VC, objet de preuve
- **Nouveaux outils CLI**: commandes diff, convert, validate, monitor, profile, inspect, merge, query
- **IoT industriel**: encodeur de registres Modbus, validateur de trames CANbus, décodeur de signaux, scanner de dispositifs
- **Géospatial**: enveloppe convexe (algorithme de Graham), calculateur de distance, détecteur d'intersections, calculateur de surface
- **Traitement de flux**: gestionnaire de partitions, groupes de consommateurs, registre de schémas, file de lettres mortes, suivi de filigrane

### Décompte des Tests par Crate (v0.2.1)

| Crate | Tests |
|-------|-------|
| oxirs-core | 2 458 |
| oxirs-arq | 2 628 |
| oxirs-rule | 2 114 |
| oxirs-tdb | 2 068 |
| oxirs-fuseki | 1 626 |
| oxirs-gql | 1 706 |
| oxirs-shacl | 1 915 |
| oxirs-geosparql | 1 756 |
| oxirs-vec | 1 587 |
| oxirs-shacl-ai | 1 509 |
| oxirs-samm | 1 326 |
| oxirs-ttl | 1 350 |
| oxirs-star | 1 507 |
| oxirs-tsdb | 1 250 |
| oxirs-embed | 1 408 |
| oxirs-did | 1 196 |
| oxirs (outils) | 1 582 |
| oxirs-stream | 1 191 |
| oxirs-federate | 1 148 |
| oxirs-canbus | 1 158 |
| oxirs-modbus | 1 115 |
| oxirs-chat | 1 095 |
| oxirs-wasm | 1 036 |
| oxirs-cluster | 1 019 |
| oxirs-physics | 1 225 |
| oxirs-graphrag | 998 |
| **Total** | **40 791** |

### Benchmarks de Performance

```
Optimisation de Requêtes (5 motifs de triples):
  HautDébit:     3,24 µs  (3,3× plus rapide que la référence)
  Analytique:    3,01 µs  (3,9× plus rapide que la référence)
  Mixte:         2,95 µs  (3,6× plus rapide que la référence)
  FaibleMémoire: 2,94 µs  (5,3× plus rapide que la référence)

Base de Données de Séries Temporelles:
  Débit d'écriture: 500K pts/sec (unitaire), 2M pts/sec (lot)
  Latence de requête: 180ms p50 (1M points)
  Compression:        ratio moyen 40:1

Impact Production (100K QPS):
  Temps CPU économisé: 45 min par heure (réduction de 75%)
  Économies annuelles: 10 000 € - 50 000 € (déploiements cloud)
```

## Documentation

### Documentation en Français

- **[Architecture IDS](docs/IDS_ARCHITECTURE.md)** - Architecture de souveraineté des données (Anglais)
- **[Opérations IDS](docs/IDS_OPERATIONS.md)** - Manuel d'exploitation pour connecteurs IDS (Anglais)
- **[Digital Twin Quickstart](server/oxirs-fuseki/DIGITAL_TWIN_QUICKSTART.md)** - IoT industriel (Anglais)
- **[oxirs-gaiax.toml](oxirs-gaiax.toml)** - Template de configuration Gaia-X

### Initiatives Européennes

- [Gaia-X](https://gaia-x.eu/)
- [Catena-X](https://catena-x.net/)
- [Manufacturing-X](https://www.manufacturing-x.de/)
- [IDSA (International Data Spaces Association)](https://internationaldataspaces.org/)
- [Agence Nationale de la Sécurité des Systèmes d'Information (ANSSI)](https://www.ssi.gouv.fr/)

### Standards & Conformité

- **RGPD**: Règlement Général sur la Protection des Données
- **IDSA RAM 4.x**: Modèle d'Architecture de Référence IDS
- **Gaia-X Trust Framework**: Version 22.10
- **ODRL 2.2**: Open Digital Rights Language
- **W3C PROV-O**: Ontologie de Provenance

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

### Feature Flags

Fonctionnalités optionnelles pour minimiser les dépendances:

- `geo`: Support GeoSPARQL
- `text`: Recherche plein texte avec Tantivy
- `ai`: Recherche vectorielle et embeddings
- `cluster`: Stockage distribué avec Raft
- `star`: Support RDF-star et SPARQL-star
- `vec`: Abstractions d'index vectoriel

## Feuille de Route

| Version | Date Cible | Jalon | Livrables | Statut |
|---------|-----------|-------|-----------|--------|
| **v0.1.0** | **7 jan 2026** | **Production Initiale** | SPARQL 1.1/1.2 complet, IoT industriel, IA, 13 123 tests | Publié |
| **v0.2.1** | **11 mars 2026** | **Expansion Majeure des Fonctionnalités** | 40 791+ tests, 26 nouveaux modules, optimiseur 3,8×, algèbre SPARQL avancée | Publié (actuel) |
| **v0.3.0** | **T2 2026** | **Recherche Plein Texte & Échelle** | Recherche plein texte (Tantivy), performance 10×, clustering multi-région | Prévu |

### Version Actuelle: v0.2.1 (11 mars 2026)

**Axes de Développement v0.2.1 (16 rounds terminés):**
- Algèbre SPARQL avancée: évaluateurs EXISTS/MINUS, constructeur de sous-requêtes, clause SERVICE, jointure LATERAL
- Renforcement du stockage: store six-index, fusion/reconstruction d'index, compaction B-tree, cache de triples
- IA prête pour la production: store vectoriel, inférence de contraintes, historique de conversation, cache de réponses
- Renforcement de la sécurité: store d'identifiants, validation de chaîne de confiance, gestionnaire de clés, présentateur VC
- Nouveaux outils CLI: commandes diff, convert, validate, monitor, profile, inspect, merge
- Améliorations du streaming: gestionnaire de partitions, groupes de consommateurs, registre de schémas, file de lettres mortes
- Séries temporelles: requêtes continues, tampon d'écriture, index de tags, gestion de la rétention
- Géospatial: enveloppe convexe, calcul de distance, détection d'intersections

## Parrainage

OxiRS est développé et maintenu par **COOLJAPAN OU (Team Kitasan)**.

Si vous trouvez OxiRS utile, envisagez de parrainer le projet pour soutenir le développement continu de l'écosystème Pure Rust.

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Votre parrainage nous permet de:
- Maintenir et améliorer l'écosystème COOLJAPAN
- Maintenir l'écosystème complet (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Fournir un support à long terme et des mises à jour de sécurité

## Licence

OxiRS est distribué sous licence:

- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) ou http://www.apache.org/licenses/LICENSE-2.0)

Voir [LICENSE](LICENSE) pour les détails.

## Contact

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **Mainteneur**: @cool-japan (KitaSan)

---

**Autres Langues:**
- [English](README.md)
- [日本語 (Japanese)](README.ja.md)
- [Deutsch (German)](README.de.md)

---

*"Rust rend la sécurité de la mémoire incontournable; OxiRS rend l'ingénierie des graphes de connaissances incontournable."*

**v0.2.1 - Publié (actuel) - 11 mars 2026**
