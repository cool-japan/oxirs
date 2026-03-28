# OxiRS

> Rust製、モジュラー型セマンティックウェブプラットフォーム - SPARQL 1.2、GraphQL、AI拡張推論対応

[![ライセンス: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![バージョン](https://img.shields.io/badge/version-0.2.3-blue)](https://github.com/cool-japan/oxirs/releases)

**ステータス**: v0.2.3 - リリース済み - 2026年3月16日

**プロダクション対応完了**: 完全なSPARQL 1.1/1.2実装、**3.8倍高速化されたオプティマイザ**、産業IoT対応、AI機能搭載。**40,791以上のテスト合格**、26クレート全体で警告ゼロ。

**v0.2.3 ハイライト（2026年3月16日）**: 16回の開発ラウンドにわたって26の新機能モジュールを追加。高度なSPARQL代数（EXISTS、MINUS、サブクエリ、サービス句）、プロダクションレベルのストレージ（6インデックスストア、インデックスマージャー/リビルダー）、AI機能（ベクターストア、制約推論、会話履歴）、セキュリティ強化（認証情報ストア、トラストチェーン検証）。

## ビジョン

OxiRSは、Apache Jena + FusekiおよびJuniperの**Rust製・JVMフリー**な代替実装を目指し、以下を提供します：

- **プロトコルの選択、ロックインなし**: 同一データセットからSPARQL 1.2とGraphQLの両方のエンドポイントを公開
- **段階的導入**: 各クレートが単独で動作、Cargoのfeatureフラグで高度な機能を選択可能
- **AI対応**: ベクトル検索、グラフ埋め込み、LLM拡張クエリとのネイティブ統合
- **シングルバイナリ**: Jena/Fusekiと機能互換を維持しながら、<50MBのフットプリント

## Society 5.0 対応

OxiRSは、**日本のSociety 5.0（ソサエティ5.0）イニシアティブ**に完全対応しています。

### 主要機能

**スマートシティ・都市計画**
- ✅ **PLATEAU統合**: 国土交通省の3D都市モデルプラットフォームに完全対応
- ✅ **NGSI-LD API**: スマートシティセンサー用のFIWARE互換コンテキストブローカー
- ✅ **ETSI GS CIM 009 V1.6.1**: 完全なNGSI-LD v1.6準拠
- ✅ **GeoSPARQL**: 日本の座標系対応（EPSG:6668 JGD2011）

**コネクテッド・インダストリーズ（製造業DX）**
- ✅ **産業IoT**: Modbus、OPC UA、CANbus（J1939）統合
- ✅ **デジタルツイン**: 工場自動化のための物理シミュレーション
- ✅ **IDS/Gaia-X**: データ主権と安全なデータ交換
- ✅ **SAMM 2.0-2.3**: 自動車産業向けセマンティックメタモデル（Catena-X）

**医療・ライフサイエンス**
- ✅ **PHR対応**: GDPR/APPI準拠の個人健康記録
- ✅ **データレジデンシー**: 日本国内のみのデータ保存ポリシー
- ✅ **検証可能資格情報**: W3C DIDによる医療ID管理

**農業・食品サプライチェーン**
- ✅ **スマート農業**: 精密農業のためのIoTセンサーネットワーク
- ✅ **サプライチェーン**: IDSコネクタ統合によるトレーサビリティ
- ✅ **気象データ**: リアルタイム環境モニタリング

**防災**
- ✅ **リアルタイム処理**: 緊急通知のためのWebSocket
- ✅ **GeoSPARQL**: 避難計画のための空間クエリ
- ✅ **センサーネットワーク**: 多元的災害監視

### クイックスタート

#### インストール

```bash
# CLIツールのインストール
cargo install oxirs --version 0.2.3

# ソースからビルド
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --workspace --release
```

#### 基本的な使い方

```bash
# ナレッジグラフの初期化（英数字、_、-のみ）
oxirs init mykg

# RDFデータのインポート（自動的に mykg/data.nq に保存）
oxirs import mykg data.ttl --format turtle

# データのクエリ（ディスクから自動読み込み）
oxirs query mykg "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# サーバーの起動
oxirs serve mykg/oxirs.toml --port 3030
```

#### Society 5.0向け設定

```bash
# Society 5.0最適化済みテンプレートを使用
cp oxirs-society5.toml oxirs.toml

# 日本向け最適化でサーバー起動
oxirs serve oxirs.toml --port 3030
```

#### PLATEAUスマートシティの例

```bash
# 東京の大気質センサーを作成（NGSI-LD）
curl -X POST http://localhost:3030/ngsi-ld/v1/entities \
  -H "Content-Type: application/ld+json" \
  -d '{
    "id": "urn:ngsi-ld:AirQualitySensor:Tokyo:Shinjuku:001",
    "type": "AirQualitySensor",
    "location": {
      "type": "GeoProperty",
      "value": {"type": "Point", "coordinates": [139.7006, 35.6895]}
    },
    "pm25": {"type": "Property", "value": 12.5, "unitCode": "GQ"},
    "temperature": {"type": "Property", "value": 22.5, "unitCode": "CEL"}
  }'

# 5km圏内のセンサーをクエリ
curl "http://localhost:3030/ngsi-ld/v1/entities?type=AirQualitySensor&georel=near;maxDistance==5000"

# SPARQLでクエリ
curl -X POST http://localhost:3030/sparql \
  -H "Content-Type: application/sparql-query" \
  -d 'SELECT ?sensor ?temp WHERE {
    ?sensor a <urn:AirQualitySensor> ;
            <urn:temperature> ?temp .
    FILTER(?temp > 20)
  }'
```

### 日本国内クラウドへのデプロイ

**AWS 東京リージョン（ap-northeast-1）**
```bash
# AWS東京リージョンに最適化設定でデプロイ
terraform apply -var="region=ap-northeast-1" -var="instance_type=t3.xlarge"
```

**Azure Japan East（東日本）**
```bash
# Azure Japan Eastにデプロイ
az deployment group create --resource-group oxirs-rg \
  --template-file deploy-azure-japan.json \
  --parameters location=japaneast vmSize=Standard_D4s_v3
```

### ユースケース

1. **PLATEAUスマートシティ**: 3D都市モデルとリアルタイムセンサー統合
2. **コネクテッドファクトリー**: Modbus/OPC UA対応のIndustry 4.0デジタルツイン
3. **医療PHR**: APPI準拠の個人健康記録
4. **スマート農業**: IoTセンサーネットワークによる精密農業
5. **防災システム**: リアルタイム監視と避難システム

### パフォーマンスベンチマーク（日本スケール）

```
東京スマートシティ展開:
  センサー数:        100,000+ IoTデバイス
  PLATEAUエリア:     東京23区
  クエリスループット:  100,000 QPS
  レイテンシ p99:     <100ms（東京→大阪間）
  データレジデンシー:  100% 日本国内保存

工場デジタルツイン:
  Modbusデバイス:     1,000+ PLC
  更新頻度:           1Hz/デバイス
  シミュレーション:   リアルタイム物理演算（SciRS2）
  稼働率:             99.9% SLA
```

## v0.2.3の新機能（2026年3月16日）

**v0.2.3リリース: 16回の開発ラウンドにわたる26の新機能モジュール**

OxiRS v0.2.3は、高度なSPARQL代数、プロダクションレベルのストレージ、AI機能、セキュリティ強化によってプラットフォームを大幅に拡張しています：

**コア機能:**
- **完全なSPARQL 1.1/1.2** - W3C準拠、高度なクエリ最適化
- **3.8倍高速化オプティマイザ** - 適応的複雑度検出による最適性能
- **高度なSPARQL代数** - EXISTS/MINUS評価器、サブクエリビルダー、サービス句、LATERALジョイン
- **産業IoT** - 時系列データ、Modbus、CANbus/J1939統合
- **AI搭載** - GraphRAG、ベクターストア、制約推論、会話履歴、熱力学シミュレーション
- **プロダクションセキュリティ** - ReBAC、OAuth2/OIDC、DID & 検証可能資格情報、トラストチェーン検証
- **ストレージ強化** - 6インデックスストア（SPO/POS/OSP/GSPO/GPOS/GOPS）、インデックスマージャー/リビルダー、B-treeコンパクション、トリプルキャッシュ
- **完全な可観測性** - Prometheusメトリクス、OpenTelemetryトレーシング
- **新CLIツール** - diff、convert、validate、monitor、profile、inspect、mergeコマンド
- **産業IoT強化** - Modbusレジスタエンコーダ、CANbusフレームバリデータ、シグナルデコーダ
- **地理空間強化** - 凸包計算（グラハムスキャン）、距離計算、交差点検出、面積計算
- **ストリーム処理** - パーティションマネージャー、コンシューマーグループ、スキーマレジストリ、デッドレターキュー
- **クラウドネイティブ** - Kubernetesオペレータ、Terraformモジュール、Docker対応

**品質メトリクス（v0.2.3）:**
- ✅ **40,791以上のテスト合格** (100%合格率、約115件スキップ)
- ✅ **コンパイル警告ゼロ** 26クレート全体
- ✅ **95%以上のテストカバレッジ**およびドキュメントカバレッジ
- ✅ **プロダクション検証済み** 産業デプロイ実績
- ✅ **26の新機能モジュール** 16回の開発ラウンドにわたって追加

## 公開済みクレート

全クレートは [crates.io](https://crates.io) に公開済みで、[docs.rs](https://docs.rs) にドキュメントがあります。

### コア

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs-core]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-core.svg)](https://crates.io/crates/oxirs-core) | [![docs.rs](https://docs.rs/oxirs-core/badge.svg)](https://docs.rs/oxirs-core) | コアRDF・SPARQL機能 |

[oxirs-core]: https://crates.io/crates/oxirs-core

### サーバー

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs-fuseki]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-fuseki.svg)](https://crates.io/crates/oxirs-fuseki) | [![docs.rs](https://docs.rs/oxirs-fuseki/badge.svg)](https://docs.rs/oxirs-fuseki) | SPARQL 1.1/1.2 HTTPサーバー |
| **[oxirs-gql]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-gql.svg)](https://crates.io/crates/oxirs-gql) | [![docs.rs](https://docs.rs/oxirs-gql/badge.svg)](https://docs.rs/oxirs-gql) | RDF向けGraphQLエンドポイント |

[oxirs-fuseki]: https://crates.io/crates/oxirs-fuseki
[oxirs-gql]: https://crates.io/crates/oxirs-gql

### エンジン

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs-arq]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-arq.svg)](https://crates.io/crates/oxirs-arq) | [![docs.rs](https://docs.rs/oxirs-arq/badge.svg)](https://docs.rs/oxirs-arq) | SPARQLクエリエンジン |
| **[oxirs-rule]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-rule.svg)](https://crates.io/crates/oxirs-rule) | [![docs.rs](https://docs.rs/oxirs-rule/badge.svg)](https://docs.rs/oxirs-rule) | ルールベース推論 |
| **[oxirs-shacl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl.svg)](https://crates.io/crates/oxirs-shacl) | [![docs.rs](https://docs.rs/oxirs-shacl/badge.svg)](https://docs.rs/oxirs-shacl) | SHACL検証 |
| **[oxirs-samm]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-samm.svg)](https://crates.io/crates/oxirs-samm) | [![docs.rs](https://docs.rs/oxirs-samm/badge.svg)](https://docs.rs/oxirs-samm) | SAMMメタモデル & AAS |
| **[oxirs-geosparql]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-geosparql.svg)](https://crates.io/crates/oxirs-geosparql) | [![docs.rs](https://docs.rs/oxirs-geosparql/badge.svg)](https://docs.rs/oxirs-geosparql) | GeoSPARQLサポート |
| **[oxirs-star]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-star.svg)](https://crates.io/crates/oxirs-star) | [![docs.rs](https://docs.rs/oxirs-star/badge.svg)](https://docs.rs/oxirs-star) | RDF-starサポート |
| **[oxirs-ttl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-ttl.svg)](https://crates.io/crates/oxirs-ttl) | [![docs.rs](https://docs.rs/oxirs-ttl/badge.svg)](https://docs.rs/oxirs-ttl) | Turtleパーサー |
| **[oxirs-vec]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-vec.svg)](https://crates.io/crates/oxirs-vec) | [![docs.rs](https://docs.rs/oxirs-vec/badge.svg)](https://docs.rs/oxirs-vec) | ベクトル検索 |

[oxirs-arq]: https://crates.io/crates/oxirs-arq
[oxirs-rule]: https://crates.io/crates/oxirs-rule
[oxirs-shacl]: https://crates.io/crates/oxirs-shacl
[oxirs-samm]: https://crates.io/crates/oxirs-samm
[oxirs-geosparql]: https://crates.io/crates/oxirs-geosparql
[oxirs-star]: https://crates.io/crates/oxirs-star
[oxirs-ttl]: https://crates.io/crates/oxirs-ttl
[oxirs-vec]: https://crates.io/crates/oxirs-vec

### ストレージ

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs-tdb]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-tdb.svg)](https://crates.io/crates/oxirs-tdb) | [![docs.rs](https://docs.rs/oxirs-tdb/badge.svg)](https://docs.rs/oxirs-tdb) | TDB2互換ストレージ |
| **[oxirs-cluster]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-cluster.svg)](https://crates.io/crates/oxirs-cluster) | [![docs.rs](https://docs.rs/oxirs-cluster/badge.svg)](https://docs.rs/oxirs-cluster) | 分散クラスタリング |

[oxirs-tdb]: https://crates.io/crates/oxirs-tdb
[oxirs-cluster]: https://crates.io/crates/oxirs-cluster

### ストリーム

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs-stream]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-stream.svg)](https://crates.io/crates/oxirs-stream) | [![docs.rs](https://docs.rs/oxirs-stream/badge.svg)](https://docs.rs/oxirs-stream) | リアルタイムストリーミング |
| **[oxirs-federate]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-federate.svg)](https://crates.io/crates/oxirs-federate) | [![docs.rs](https://docs.rs/oxirs-federate/badge.svg)](https://docs.rs/oxirs-federate) | フェデレーションクエリ |

[oxirs-stream]: https://crates.io/crates/oxirs-stream
[oxirs-federate]: https://crates.io/crates/oxirs-federate

### AI

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs-embed]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-embed.svg)](https://crates.io/crates/oxirs-embed) | [![docs.rs](https://docs.rs/oxirs-embed/badge.svg)](https://docs.rs/oxirs-embed) | ナレッジグラフ埋め込み・ベクターストア |
| **[oxirs-shacl-ai]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl-ai.svg)](https://crates.io/crates/oxirs-shacl-ai) | [![docs.rs](https://docs.rs/oxirs-shacl-ai/badge.svg)](https://docs.rs/oxirs-shacl-ai) | AI搭載SHACL制約推論 |
| **[oxirs-chat]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-chat.svg)](https://crates.io/crates/oxirs-chat) | [![docs.rs](https://docs.rs/oxirs-chat/badge.svg)](https://docs.rs/oxirs-chat) | 会話履歴付きRAGチャットAPI |
| **[oxirs-physics]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-physics.svg)](https://crates.io/crates/oxirs-physics) | [![docs.rs](https://docs.rs/oxirs-physics/badge.svg)](https://docs.rs/oxirs-physics) | 物理情報デジタルツイン推論 |
| **[oxirs-graphrag]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-graphrag.svg)](https://crates.io/crates/oxirs-graphrag) | [![docs.rs](https://docs.rs/oxirs-graphrag/badge.svg)](https://docs.rs/oxirs-graphrag) | GraphRAGハイブリッド検索（ベクター×グラフ） |

[oxirs-embed]: https://crates.io/crates/oxirs-embed
[oxirs-shacl-ai]: https://crates.io/crates/oxirs-shacl-ai
[oxirs-chat]: https://crates.io/crates/oxirs-chat
[oxirs-physics]: https://crates.io/crates/oxirs-physics
[oxirs-graphrag]: https://crates.io/crates/oxirs-graphrag

### セキュリティ

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs-did]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-did.svg)](https://crates.io/crates/oxirs-did) | [![docs.rs](https://docs.rs/oxirs-did/badge.svg)](https://docs.rs/oxirs-did) | DID & 検証可能資格情報 |

[oxirs-did]: https://crates.io/crates/oxirs-did

### プラットフォーム

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs-wasm]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-wasm.svg)](https://crates.io/crates/oxirs-wasm) | [![docs.rs](https://docs.rs/oxirs-wasm/badge.svg)](https://docs.rs/oxirs-wasm) | WASMブラウザ/エッジデプロイ |

[oxirs-wasm]: https://crates.io/crates/oxirs-wasm

### ツール

| クレート | バージョン | ドキュメント | 説明 |
|---------|-----------|-------------|------|
| **[oxirs (CLI)]** | [![Crates.io](https://img.shields.io/crates/v/oxirs.svg)](https://crates.io/crates/oxirs) | [![docs.rs](https://docs.rs/oxirs/badge.svg)](https://docs.rs/oxirs) | CLIツール |

[oxirs (CLI)]: https://crates.io/crates/oxirs

## アーキテクチャ

```
oxirs/                  # Cargoワークスペースルート
├─ core/                # 基盤モジュール
│  └─ oxirs-core
├─ server/              # ネットワークフロントエンド
│  ├─ oxirs-fuseki      # SPARQL 1.1/1.2 HTTPサーバー
│  └─ oxirs-gql         # GraphQLファサード
├─ engine/              # クエリ、更新、推論
│  ├─ oxirs-arq         # Jenaスタイル代数 + 拡張ポイント
│  ├─ oxirs-rule        # 前方/後方推論エンジン（RDFS/OWL/SWRL）
│  ├─ oxirs-samm        # SAMMメタモデル + AAS統合（Industry 4.0）
│  ├─ oxirs-geosparql   # GeoSPARQL空間クエリ
│  ├─ oxirs-shacl       # SHACL Core + SHACL-SPARQL検証
│  ├─ oxirs-star        # RDF-star / SPARQL-star文法サポート
│  ├─ oxirs-ttl         # Turtle/TriGパーサー
│  └─ oxirs-vec         # ベクトルインデックス抽象化
├─ storage/
│  ├─ oxirs-tdb         # MVCCレイヤー（TDB2互換）
│  └─ oxirs-cluster     # Raft分散データセット
├─ stream/              # リアルタイムと連携
│  ├─ oxirs-stream      # Kafka/NATS I/O、RDF Patch
│  └─ oxirs-federate    # SERVICEプランナー、GraphQLステッチング
├─ ai/
│  ├─ oxirs-embed       # KG埋め込み（TransE、ComplEx...）
│  ├─ oxirs-shacl-ai    # シェイプ誘導とデータ修復提案
│  ├─ oxirs-chat        # RAGチャットAPI（LLM + SPARQL）
│  ├─ oxirs-physics     # 物理情報デジタルツイン
│  └─ oxirs-graphrag    # GraphRAGハイブリッド検索
├─ security/
│  └─ oxirs-did         # W3C DID & 検証可能資格情報
├─ platforms/
│  └─ oxirs-wasm        # WebAssemblyブラウザ/エッジデプロイ
└─ tools/
    ├─ oxirs             # CLI（インポート、エクスポート、ベンチマーク）
    └─ benchmarks/       # SP2Bench、WatDiv、LDBC SGS
```

## 機能マトリクス（v0.2.3）

| 機能 | OxiRSクレート | ステータス | Jena/Fuseki互換性 |
|------|---------------|-----------|-------------------|
| **コアRDF & SPARQL** | | | |
| RDF 1.2 & 7フォーマット | `oxirs-core` | ✅ 安定版（2458テスト） | ✅ |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ 安定版（1626 + 2628テスト） | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (`star` flag) | ✅ 安定版 | 🔸 |
| 高度なSPARQL代数（EXISTS/MINUS/サブクエリ） | `oxirs-arq` | ✅ 安定版 | ✅ |
| 永続ストレージ（N-Quads） | `oxirs-core` | ✅ 安定版 | ✅ |
| **セマンティックウェブ拡張** | | | |
| RDF-star パース/シリアライズ | `oxirs-star` | ✅ 安定版（1507テスト） | 🔸 |
| SHACL Core+API（W3C準拠） | `oxirs-shacl` | ✅ 安定版（1915テスト、W3C 27/27） | ✅ |
| ルール推論（RDFS/OWL 2 DL） | `oxirs-rule` | ✅ 安定版（2114テスト） | ✅ |
| SAMM 2.0-2.3 & AAS（Industry 4.0） | `oxirs-samm` | ✅ 安定版（1326テスト、16ジェネレータ） | ❌ |
| **クエリ & 連携** | | | |
| GraphQL API | `oxirs-gql` | ✅ 安定版（1706テスト） | ❌ |
| SPARQL連携（SERVICE） | `oxirs-federate` | ✅ 安定版（1148テスト、2PC） | ✅ |
| フェデレーション認証 | `oxirs-federate` | ✅ 安定版（OAuth2/SAML/JWT） | 🔸 |
| **リアルタイム & ストリーミング** | | | |
| ストリーム処理（Kafka/NATS） | `oxirs-stream` | ✅ 安定版（1191テスト、SIMD） | 🔸 |
| RDF Patch & SPARQL Updateデルタ | `oxirs-stream` | ✅ 安定版 | 🔸 |
| **検索 & 地理空間** | | | |
| 全文検索（`text:`） | `oxirs-textsearch` | ⏳ 計画中 | ✅ |
| GeoSPARQL（OGC 1.1） | `oxirs-geosparql` | ✅ 安定版（1756テスト） | ✅ |
| ベクトル検索/埋め込み | `oxirs-vec`（1587テスト）、`oxirs-embed`（1408テスト） | ✅ 安定版 | ❌ |
| **ストレージ & 分散** | | | |
| TDB2互換ストレージ（6インデックス） | `oxirs-tdb` | ✅ 安定版（2068テスト） | ✅ |
| 分散/HAストア（Raft） | `oxirs-cluster` | ✅ 安定版（1019テスト） | 🔸 |
| 時系列データベース | `oxirs-tsdb` | ✅ 安定版（1250テスト） | ❌ |
| **AI & 高度機能** | | | |
| RAGチャットAPI（LLM統合） | `oxirs-chat` | ✅ 安定版（1095テスト） | ❌ |
| AI搭載SHACL制約推論 | `oxirs-shacl-ai` | ✅ 安定版（1509テスト） | ❌ |
| GraphRAGハイブリッド検索 | `oxirs-graphrag` | ✅ 安定版（998テスト） | ❌ |
| 物理情報デジタルツイン | `oxirs-physics` | ✅ 安定版（1225テスト） | ❌ |
| ナレッジグラフ埋め込み（TransE等） | `oxirs-embed` | ✅ 安定版（1408テスト） | ❌ |
| **セキュリティ & 信頼** | | | |
| W3C DID & 検証可能資格情報 | `oxirs-did` | ✅ 安定版（1196テスト） | ❌ |
| トラストチェーン検証 | `oxirs-did` | ✅ 安定版 | ❌ |
| 署名付きRDFグラフ（RDFC-1.0） | `oxirs-did` | ✅ 安定版 | ❌ |
| Ed25519暗号証明 | `oxirs-did` | ✅ 安定版 | ❌ |
| **セキュリティ & 認可** | | | |
| ReBAC（関係ベースアクセス制御） | `oxirs-fuseki` | ✅ 安定版 | ❌ |
| グラフレベル認可 | `oxirs-fuseki` | ✅ 安定版 | ❌ |
| SPARQLベース認可ストレージ | `oxirs-fuseki` | ✅ 安定版 | ❌ |
| OAuth2/OIDC/SAML認証 | `oxirs-fuseki` | ✅ 安定版 | 🔸 |
| **ブラウザ & エッジデプロイ** | | | |
| WebAssembly（WASM）バインディング | `oxirs-wasm` | ✅ 安定版（1036テスト） | ❌ |
| ブラウザRDF/SPARQL実行 | `oxirs-wasm` | ✅ 安定版 | ❌ |
| TypeScript型定義 | `oxirs-wasm` | ✅ 安定版 | ❌ |
| Cloudflare Workers / Denoサポート | `oxirs-wasm` | ✅ 安定版 | ❌ |
| **産業IoT** | | | |
| Modbus TCP/RTUプロトコル | `oxirs-modbus` | ✅ 安定版（1115テスト） | ❌ |
| CANbus / J1939プロトコル | `oxirs-canbus` | ✅ 安定版（1158テスト） | ❌ |

**凡例:**
- ✅ 安定版: プロダクション対応、包括的テスト、API安定性保証
- ⏳ 計画中: 未実装
- 🔸 Jenaでは部分的/プラグイン対応

**品質メトリクス（v0.2.3）:**
- **40,791テスト合格** (100%合格率、約115件スキップ)
- **コンパイル警告ゼロ** (`-D warnings` 強制)
- **95%以上のテストカバレッジ** 全26モジュール
- **95%以上のドキュメントカバレッジ**
- **全統合テスト合格**
- **プロダクションレベルのセキュリティ監査完了**
- **CUDA GPU対応** AIアクセラレーション用
- **3.8倍高速クエリ最適化** 適応的複雑度検出による
- **26の新機能モジュール** v0.2.3（16回の開発ラウンド）で追加

## リリースノート（v0.2.3）

完全なノートは [CHANGELOG.md](CHANGELOG.md) にあります。

### ハイライト（2026年3月16日）
- **40,791以上のテスト合格** 全26クレートにわたって
- **26の新機能モジュール** 16回の開発ラウンドにわたって全26クレートに追加
- **高度なSPARQL代数**: EXISTS評価器、MINUS評価器、サブクエリビルダー、サービス句ハンドラー
- **ストレージ強化**: 6インデックスストア（SPO/POS/OSP/GSPO/GPOS/GOPS）、インデックスマージャー/リビルダー、B-treeコンパクション
- **AIプロダクション対応**: ベクターストア、制約推論、会話履歴、応答キャッシュ、リランカー
- **セキュリティ強化**: 認証情報ストア、トラストチェーン検証、鍵管理、VC提示者、証明目的
- **新CLIツール**: diff、convert、validate、monitor、profile、inspect、merge、queryコマンド
- **産業IoT**: Modbusレジスタエンコーダ、CANbusフレームバリデータ、シグナルデコーダ、デバイススキャナー
- **地理空間**: 凸包（グラハムスキャン）、距離計算、交差点検出、面積計算
- **ストリーム処理**: パーティションマネージャー、コンシューマーグループ、スキーマレジストリ、デッドレターキュー、ウォーターマーク追跡

### クレート別テスト件数（v0.2.3）

| クレート | テスト数 |
|---------|---------|
| oxirs-core | 2458 |
| oxirs-arq | 2628 |
| oxirs-rule | 2114 |
| oxirs-tdb | 2068 |
| oxirs-fuseki | 1626 |
| oxirs-gql | 1706 |
| oxirs-shacl | 1915 |
| oxirs-geosparql | 1756 |
| oxirs-vec | 1587 |
| oxirs-shacl-ai | 1509 |
| oxirs-samm | 1326 |
| oxirs-ttl | 1350 |
| oxirs-star | 1507 |
| oxirs-tsdb | 1250 |
| oxirs-embed | 1408 |
| oxirs-did | 1196 |
| oxirs（ツール） | 1582 |
| oxirs-stream | 1191 |
| oxirs-federate | 1148 |
| oxirs-canbus | 1158 |
| oxirs-modbus | 1115 |
| oxirs-chat | 1095 |
| oxirs-wasm | 1036 |
| oxirs-cluster | 1019 |
| oxirs-physics | 1225 |
| oxirs-graphrag | 998 |
| **合計** | **40,791** |

### パフォーマンスベンチマーク

```
クエリ最適化（5トリプルパターン）:
  HighThroughput:  3.24 µs  (ベースラインより3.3倍高速)
  Analytical:      3.01 µs  (ベースラインより3.9倍高速)
  Mixed:           2.95 µs  (ベースラインより3.6倍高速)
  LowMemory:       2.94 µs  (ベースラインより5.3倍高速)

時系列データベース:
  書き込みスループット: 500K pts/sec（単体）、2M pts/sec（バッチ）
  クエリレイテンシ:     180ms p50（100万ポイント）
  圧縮率:              平均40:1

プロダクション影響（100K QPS）:
  CPU時間節約: 毎時45分（75%削減）
  年間コスト削減: $10,000 - $50,000（クラウドデプロイ）
```

## ドキュメント

### 日本語ドキュメント

- **[Society 5.0統合ガイド](docs/SOCIETY5_INTEGRATION.md)** - 完全な日英バイリンガルガイド
- **[IDSアーキテクチャ](docs/IDS_ARCHITECTURE.md)** - データ主権アーキテクチャ
- **[デジタルツインクイックスタート](server/oxirs-fuseki/DIGITAL_TWIN_QUICKSTART.md)** - 産業IoT例
- **[oxirs-society5.toml](oxirs-society5.toml)** - プロダクション設定テンプレート

### 政府施策

- [Society 5.0（内閣府）](https://www8.cao.go.jp/cstp/society5_0/)
- [PLATEAU（国土交通省）](https://www.mlit.go.jp/plateau/)
- [デジタル庁](https://www.digital.go.jp/)

### 標準規格準拠

- ✅ **APPI**: 個人情報保護法
- ✅ **JISC**: 日本工業標準調査会
- ✅ **METI**: コネクテッド・インダストリーズ
- ✅ **MIC**: スマートシティリファレンスアーキテクチャ

## 使用例

### データセット設定（TOML）

```toml
[dataset.mykg]
type      = "tdb2"
location  = "/data"
text      = { enabled = true, analyzer = "english" }
shacl     = ["./shapes/person.ttl"]

# ReBAC認可（オプション）
[security.policy_engine]
mode = "Combined"  # RbacOnly | RebacOnly | Combined | Both

[security.rebac]
backend = "InMemory"  # InMemory | RdfNative
namespace = "http://oxirs.org/auth#"
inference_enabled = true

[[security.rebac.initial_relationships]]
subject = "user:alice"
relation = "owner"
object = "dataset:mykg"
```

### GraphQLクエリ（自動生成）

```graphql
query {
  Person(where: {familyName: "Yamada"}) {
    givenName
    homepage
    knows(limit: 5) { givenName }
  }
}
```

### ベクター類似度SPARQLサービス（オプトインAI）

```sparql
SELECT ?s ?score WHERE {
  SERVICE <vec:similar ( "セマンティックウェブのLLM埋め込み" 0.8 )> {
    ?s ?score .
  }
}
```

### 物理シミュレーション（SciRS2統合）

```rust
use oxirs_physics::simulation::SimulationOrchestrator;

let mut orchestrator = SimulationOrchestrator::new();
orchestrator.register("thermal", Arc::new(SciRS2ThermalSimulation::default()));

// RDFからパラメータを抽出し、シミュレーションを実行し、結果をRDFに書き戻す
let result = orchestrator.execute_workflow(
    "urn:battery:cell:001",
    "thermal"
).await?;
```

## 開発

### 前提条件

- Rust 1.70+（MSRV）
- オプション: コンテナデプロイ用Docker

### ビルド

```bash
# リポジトリのクローン
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# 全クレートのビルド
cargo build --workspace

# テスト実行
cargo nextest run --no-fail-fast

# 全機能を有効にしてビルド
cargo build --workspace --all-features
```

### フィーチャーフラグ

依存関係を最小限に保つためのオプションフィーチャー：

- `geo`: GeoSPARQLサポート
- `text`: Tantivyによる全文検索
- `ai`: ベクトル検索と埋め込み
- `cluster`: Raftによる分散ストレージ
- `star`: RDF-starとSPARQL-starサポート
- `vec`: ベクトルインデックス抽象化

## スポンサーシップ

OxiRSは **COOLJAPAN OU (Team Kitasan)** によって開発・保守されています。

OxiRSが役立つと感じたら、Pure Rustエコシステムの継続的な開発を支援するためにスポンサーをご検討ください。

[![スポンサー](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

スポンサーシップは以下の活動を支援します：
- COOLJAPANエコシステムの維持と改善
- エコシステム全体（OxiBLAS、OxiFFT、SciRS2等）の100% Pure Rust維持
- 長期サポートとセキュリティアップデートの提供

## ロードマップ

| バージョン | 予定日 | マイルストーン | 成果物 | ステータス |
|-----------|--------|---------------|--------|-----------|
| **v0.1.0** | **✅ 2026年1月7日** | **初回プロダクション** | 完全なSPARQL 1.1/1.2、産業IoT、AI機能、13,123テスト | ✅ リリース済 |
| **v0.2.3** | **✅ 2026年3月16日** | **深度機能拡張** | 40,791以上のテスト、26の新モジュール、3.8倍高速オプティマイザ、高度なSPARQL代数、AIプロダクション対応 | ✅ リリース済（現行） |
| **v0.3.0** | **2026年Q2** | **全文検索 & スケール** | 全文検索（Tantivy）、10倍パフォーマンス、マルチリージョンクラスタリング | 計画中 |

### 現行リリース: v0.2.3（2026年3月16日）

**v0.2.3 フォーカスエリア（16ラウンド完了）:**
- 高度なSPARQL代数: EXISTS/MINUS評価器、サブクエリビルダー、サービス句、LATERALジョイン
- ストレージ強化: 6インデックスストア、インデックスマージャー/リビルダー、B-treeコンパクション、トリプルキャッシュ
- AIプロダクション対応: ベクターストア、制約推論、会話履歴、応答キャッシュ
- セキュリティ強化: 認証情報ストア、トラストチェーン検証、鍵管理、VC提示者
- 新CLIツール: diff、convert、validate、monitor、profile、inspect、mergeコマンド
- ストリーム強化: パーティションマネージャー、コンシューマーグループ、スキーマレジストリ、デッドレターキュー
- 時系列: 継続的クエリ、書き込みバッファ、タグインデックス、リテンション管理

## ライセンス

OxiRSは以下のライセンスで提供されます：

- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) または http://www.apache.org/licenses/LICENSE-2.0)

詳細は[LICENSE](LICENSE)を参照してください。

## お問い合わせ

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **メンテナ**: @cool-japan（KitaSan）

---

**その他の言語:**
- [English](README.md)
- [Deutsch (German)](README.de.md)
- [Français (French)](README.fr.md)

---

*"Rustはメモリ安全性を当たり前にした。OxiRSはナレッジグラフエンジニアリングを当たり前にする。"*

**v0.2.3 - リリース済み - 2026年3月16日**
