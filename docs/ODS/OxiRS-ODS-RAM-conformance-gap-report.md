# OxiRS × Open Data Spaces (ODS-RAM) 適合性・ギャップ分析レポート

**対象**: ODS-RAM V2 / Open Data Spaces Protocols (ODP) — L4 セマンティクスレイヤ
**評価対象実装**: OxiRS v0.3.1 (2026-06-06, Apache-2.0, Pure Rust, 26 crates)
**作成**: COOLJAPAN OÜ (Team Kitasan)
**版**: 0.1 (draft)

---

## 0. 本レポートの位置づけと限界（先に明示）

本レポートは、**公開されている ODS-RAM / ODP 仕様と、OxiRS の公開ドキュメント・ソースに基づく机上の対応付け**である。

- **適合試験（conformance test）は未実施**である。ODS 側に公式の L4 適合試験スイートが公開されていないため、現時点で「適合」を主張することはできない。本レポートが用いる ✅ は「仕様が要求する機能に対応するコンポーネントが存在し、テストが通っている」という意味であり、**第三者検証を経たものではない**。
- 依存関係の完全な棚卸し（SBOM）は本レポートに添付していない。要請があれば提出する。
- 目的は売り込みではなく、**「L4 に Pure Rust の実装候補が存在すること」の情報提供**と、**仕様側に見つかった 1 点のギャップの指摘**である。

---

## 1. 要旨

1. **ODS-RAM の L4 セマンティクスレイヤは、事実上 RDF/SPARQL/SAMM の仕様である。** ODP「Metadata Exchange (L4)」の規範要件は、メタデータを RDF で表現すること（SHALL）、SAMM 等によるインタフェース意味定義を含むこと（SHALL）、Metadata Endpoint が SPARQL エンドポイントを含むこと、RDF Patch による差分同期（MAY）を明記している。

2. **OxiRS はこれらを既に実装している。** SPARQL 1.1/1.2、RDF 1.2、SAMM 2.0–2.3 + AAS（16 ジェネレータ）、RDF Patch、GeoSPARQL (OGC 1.1)、SHACL（W3C 27/27）、DID/VC（署名付き RDF, RDFC-1.0, Ed25519）を 26 クレート・約 43,500 テストで提供する。**Apache Jena には SAMM/AAS 対応が存在しない。**

3. **一方、L4 の公式参照実装（`open-dataspaces/L4-crawler-service`）は Python 製のクローラサービスであり、GitHub スター数は 1 である。** Organization 全体（20 リポジトリ）の言語は Java / JavaScript / Python / Shell / TypeScript / Vue であり、**Rust 実装は存在しない。** 最多スターのリポジトリ（`SDK-for-semantics`）でも 2 である。

4. **仕様側に 1 点、埋まっていない穴がある。** ODP L4 は「差分管理に lineage・タイムスタンプ・バージョン管理を用いるべき（SHOULD）」としながら、**その語彙を指定していない。** データスペースの本質が来歴の相互運用である以上、ここは W3C **PROV-O** を採るのが自然であり、本レポートはその採用を提案する。同時に、これは OxiRS 側にも欠けているクレートである（後述）。

5. **Pure Rust であることは、この文脈では性能の話ではなく主権の話である。** OxiRS v0.3.1 は Pure-Rust 移行を完了し、**既定ビルドで `ring` / `aws-lc-sys` などの C/アセンブリ暗号を一切リンクしない**（`oxicrypto` / `oxitls`）。JVM も不要で、単一静的バイナリ（< 50MB）として動作する。データ主権を掲げる枠組みにおいて、**ホスティングの主権ではなく供給網の主権**を技術的に裏づけられる実装は、現状ほかに無い。

---

## 2. 背景：ODS-RAM の構造

ODS-RAM V2 は、4 つの疎結合レイヤで構成される（各レイヤは detachable で、ドメインの成熟度に応じて選択的に採用できる）。

| レイヤ | 扱う問題 | 世界仮定 |
|---|---|---|
| **L1 Data Layer** | Usage Control / Data Tampering / Data Quality | CWA（閉世界） |
| **L2 Transaction Layer** | Modal / Query / Protocol | — |
| **L3 Identity Layer** | Authentication / Authorization（Verifiable Credentials） | — |
| **L4 Semantics Layer** | **Addressability / Semantics（Metadata）** | **OWA（開世界）** |

ここで注目すべきは、ODS-RAM が 2 つのプロダクト形態を定義していることである。

- **Data Product** = L3 → L1
- **Ontology Product** = **L4 → L2**

**OxiRS は Ontology Product の実装である。** 開発者向けガイドブックにも「System Build and Operations Setup Procedures (Ontology as a Product)」という独立章が置かれている。

なお L4 が **OWA（開世界仮定）** を採ることが明記されている点は重要である。OWA は RDF のセマンティクスそのものであり、L4 の仕様は「RDF」と書かずに RDF を要求している。

---

## 3. 現行の参照実装（`github.com/open-dataspaces`, 2026-07 時点）

IPA DADC が公式 GitHub Organization として公開。全 20 リポジトリ、ライセンスは MIT。

| リポジトリ | レイヤ | 言語 | ★ |
|---|---|---|---|
| `SDK-for-semantics` | SDK | Python | 2 |
| `L4-crawler-service`（分散型セマンティクス情報管理 Crawler Service） | **L4** | **Python** | 1 |
| `SDK-docker-compose` | SDK | Shell | 1 |
| `SDK-mock-server` | SDK | — | 1 |
| `L2-dp-webapi` | L2 | Java | 0 |
| `L3-identity-component` | L3 | Java | 0 |
| `SDK-client-library-java` / `-python` | SDK | Java / Python | 0 |
| `CF-Notifier` | CF | Python | 0 |
| `DCS-Payment` | DCS | Python | 0 |

**言語フィルタに Rust は存在しない。**

これは批判ではなく事実の記録である。DADC が明示している目的は「特定個社へのベンダーロックインを回避し、国際的な相互運用性を担保する」ことであり、**実装言語スタックの多様性はその目的に直接資する**。本レポートはその文脈で提出する。

---

## 4. 適合性マトリクス（1）: ODP — Metadata Exchange (L4)

出典: `open-dataspaces.gitbook.io/ods-docs/odp/fundamental-protocols/metadata-exchange-l4/protocol.md`

| # | ODP 規範要件（要約） | Lv | 対応 OxiRS クレート | 状況 |
|---|---|---|---|---|
| 1 | メタデータは **RDF** で表現されること | SHALL | `oxirs-core`, `oxirs-ttl`（RDF 1.2, 7 形式） | ✅ |
| 2 | メタデータは **SAMM 等**によるインタフェース意味定義を、グローバルに解決可能な形で含むこと | SHALL | **`oxirs-samm`（SAMM 2.0–2.3 + AAS, 16 ジェネレータ）** | ✅ **※Jena は非対応** |
| 3 | 宛先情報（エンドポイント）を含むこと | SHALL | `oxirs-core`, `oxirs-fuseki` | ✅ |
| 4 | Metadata Endpoint は **SPARQL エンドポイント**を含む | 構成要件 | `oxirs-fuseki`（SPARQL 1.1/1.2 HTTP, Fuseki 互換） | ✅ **SPARQL 1.2** |
| 5 | **JSON-LD** で送受信されるべき | SHOULD | `oxirs-core` | ◐ **要確認**（JSON-LD シリアライザの網羅性を検証すること） |
| 6 | 実世界エンティティとの関連付け | SHOULD | `oxirs-core`, `oxirs-geosparql` | ✅ |
| 7 | 他プロバイダのエンティティへの参照関係 | SHOULD | `oxirs-core`, `oxirs-federate` | ✅ |
| 8 | **RDF Patch** 等による差分管理（結果整合性） | MAY | `oxirs-stream`（RDF Patch & SPARQL Update delta） | ✅ |
| 9 | **lineage / timestamp / version** による差分管理 | SHOULD | **語彙が仕様側で未指定** | ⚠️ **§6 参照** |
| 10 | **L3（Identity & Trust）と統合可能であること** | SHALL | `oxirs-did`（W3C DID/VC, 署名付き RDF グラフ RDFC-1.0, Ed25519 証明, トラストチェーン検証） | ✅ |
| 11 | メタデータ自体へのアクセス制御（必要時） | SHALL | `oxirs-fuseki`（ReBAC, グラフ単位認可, OAuth2/OIDC/SAML） | ✅ |
| 12 | RDF モデルは ODS セマンティクスポリシーに準拠すること | SHALL | — | ◐ **ポリシー文書の所在を確認したい** |

**`ods:` 名前空間 (`https://github.com/ODS-DFS-L4/ods/`) の語彙**（`ods:DomainApp` / `ods:BaseEndpoint` / `ods:SparqlEndpoint` / `ods:hasSparqlEndpoint` / `ods:accessURL` 等）は OxiRS に未実装。**独立クレート `oxirs-ods` として実装可能**（見積: 小）。

特筆すべきは要件 #4 で、**ODS のメタデータモデルは SPARQL エンドポイントを一級市民として持つ**（`ods:SparqlEndpoint` クラスが存在する）。これは L4 が SPARQL を前提に設計されていることの直接的な証拠である。

---

## 5. 適合性マトリクス（2）: ODP — Discovery and Search (L4)

出典: `open-dataspaces.gitbook.io/ods-docs/odp/fundamental-protocols/discovery-and-search-l4/protocol.md`

| # | ODP 規範要件（要約） | Lv | 対応 OxiRS クレート | 状況 |
|---|---|---|---|---|
| 1 | Discovery Service は登録 API（register）を提供 | SHALL | — | ✗ **未実装**（ODS 固有 API） |
| 2 | 属性キーでエンドポイントを返す find API | SHALL | `oxirs-arq` / `oxirs-fuseki` 上に実装可 | ◐ **薄いラッパで実現可** |
| 3 | 検索結果はエンドポイントを含むメタデータを返す | SHALL | `oxirs-arq`（CONSTRUCT/DESCRIBE） | ◐ |
| 4 | **キーワード検索**をサポートすべき | SHOULD | 全文検索（Tantivy）, `oxirs-vec` | ◐ |
| 5 | **地理空間検索**をサポートしてもよい | MAY | **`oxirs-geosparql`（GeoSPARQL OGC 1.1, 1,713 テスト）** | ✅ **MAY を大幅に上回る** |
| 6 | TTL / keep-alive によるソフトステート管理 | SHOULD | — | ✗ **未実装** |
| 7 | 結果整合性で可 | MAY | `oxirs-cluster`（Raft）, `oxirs-stream` | ✅ |
| 8 | Discovery Service 自身のメタデータも SAMM 等で記述してよい | MAY | `oxirs-samm` | ✅ |
| 9 | **SAMM のサービス定義から Discovery Service のコンポーネントを生成してよい** | MAY | **`oxirs-samm`（16 ジェネレータ）** | ◐ **最も有望な適合点** |
| 10 | Discovery Finder（Discovery Service の解決）API | SHALL | — | ✗ **未実装** |

**評価**: Discovery and Search は、OxiRS の既存機能の上に **API 層を薄く載せるだけで実装可能**である。要件 #9（SAMM 定義からのコンポーネント自動生成）は、`oxirs-samm` が既に 16 種のジェネレータを持つため、**仕様が MAY として想定した機能を、既存資産で直接満たせる**数少ない例である。

---

## 6. 【仕様側への提案】lineage 語彙の未指定について

ODP L4 Metadata Exchange の規範要件に、以下がある（要約）:

> 差分管理の機構（RDF Patch 等）を提供してもよい。結果整合性を前提とする。必要に応じて **lineage、タイムスタンプ、バージョン管理**を用いて差分を管理すべき（SHOULD）。

**この要件は語彙を指定していない。** 一方 ODS の目的は「企業・業界・国境を越えたデータ連携」であり、Catena-X 側の要求（EU バッテリー規則、ESPR / Digital Product Passport）は、本質的に**「この値はどこから来て、誰が、何から、いつ導出したか」の相互運用可能な記録**である。

**lineage の語彙が相互運用されないなら、データスペース間の来歴は相互運用されない。**

### 提案

**W3C PROV-O (Recommendation, 2013) の採用を提案する。**

- `prov:Entity` / `prov:Activity` / `prov:Agent`、`prov:wasDerivedFrom` / `wasGeneratedBy` / `wasAttributedTo` / `used`、および必要に応じて qualified 形式（`prov:Derivation`）
- **名前付きグラフ**または **RDF-star** による、トリプル単位・グラフ単位の来歴付与
- ODS L3（DID/VC）との合成: **PROV の導出チェーンを Verifiable Credential で署名する**ことで、暗号学的に検証可能な来歴記録になる。ODP が L3 統合を SHALL としている以上、この合成は仕様の意図に沿う。

PROV-O はロイヤリティフリーの W3C 勧告であり、13 年間安定している。新規語彙を発明する必要はない。

**OxiRS 側の対応**: 現在 `oxirs-prov` は存在しない。**これは本レポートが自認する最大のギャップである。** L3（`oxirs-did`）が「誰が署名したか」を、SHACL（`oxirs-shacl`）が「形が正しいか」を担っているが、「どう導出されたか」を担うクレートが無い。仕様側でこの語彙が確定すれば、実装は即座に追随する。

---

## 7. OxiRS 側のギャップ（自認）

| # | 不足項目 | 影響レイヤ | 見積 |
|---|---|---|---|
| G1 | `ods:` 名前空間の語彙・オントロジー実装（`oxirs-ods`） | L4 | 小 |
| G2 | **PROV 語彙（`oxirs-prov`）** — §6 | L4 / L1 | 中 |
| G3 | Discovery Finder / Discovery Service の register / find API | L4 | 中 |
| G4 | TTL / keep-alive ソフトステート管理 | L4 | 小 |
| G5 | ODS SDK（Onboarding / Semantics）との相互運用確認 | SDK | 要調査 |
| G6 | JSON-LD シリアライズの網羅性検証 | L4 | 小 |
| G7 | MCP サーバ面（エージェントからの L4 アクセス） | — | 中（§8 参照） |
| G8 | **L4 適合試験スイートそのものが存在しない**（ODS 側・実装側とも） | — | 共同課題 |

G8 は本質的な論点である。**適合を主張する手段が現在ない。** 実装側から適合試験スイートを提供する用意がある。

---

## 8. Pure Rust 実装がもたらす固有の価値

ODS / ウラノス・エコシステムの目的が **データ主権**である以上、以下は性能の議論ではなく主権の議論である。

### 8.1 供給網の主権（supply-chain sovereignty）

OxiRS **v0.3.1 で COOLJAPAN Pure-Rust 移行が完了**した。

- 圧縮: brotli / snap / flate2 → `oxiarc`（Pure Rust）
- 暗号: `ring` → `oxicrypto`（Pure Rust）
- TLS: Pure Rust `oxitls` プロバイダ
- **既定の `cargo build` は `ring` / `aws-lc-sys` などの C/アセンブリ暗号を一切リンクしない**

Gaia-X が最高主権ティア（Label Level 3）で問うているのは「域外法（US Cloud Act 等）の適用を受けないこと」である。しかし**現行のデータスペース実装は、その暗号処理を C 製の外部ライブラリに委ねている。** 「EU のデータセンターで動く Java アプリが、米国由来の C 暗号ライブラリを呼ぶ」構成は、ホスティングの主権であって供給網の主権ではない。

**OxiRS は、暗号プリミティブまで含めて全ソースが Rust であり、全依存が監査可能である。** これは現在ほかに存在しない性質である。

さらに v0.3.1 は **FIPS 140-2 feature gate**（`oxirs-fuseki`, `oxirs-did`）を備え、認証適合の経路を確保している。

### 8.2 オンボーディングコストの構造的削減

Gaia-X / Catena-X の最大の実務的障害は、**中小サプライヤがコネクタ・ID 管理・カタログを自前で運用できない**ことである（Gaia-X 自身がこれをコスト問題として明言している）。結果、「主権的・分散的」なデータスペースが、中間業者のマネージドサービスとして配布されている。

- **JVM 不要・単一静的バイナリ（< 50MB）**
- `cargo install oxirs` → `oxirs serve` で起動
- Kubernetes / Terraform / Docker 対応（必要な場合のみ）

これは思想の問題ではなく、**裾野のサプライヤが実際にデータスペースに参加できるかどうか**の問題である。

### 8.3 エッジ / ブラウザ実行

ODP は Metadata Client を「ブラウザの JavaScript ライブラリ」として想定している。`oxirs-wasm`（858 テスト、TypeScript 型定義、Cloudflare Workers / Deno 対応）は、**RDF/SPARQL 処理をブラウザ内で完結させる**ことを可能にする。

### 8.4 ベンダーロックイン回避への直接的寄与

DADC は「特定個社へのベンダーロックインを回避」を明示的な目的として掲げている。現行 OSS の言語スタックは Java / Python に集中している。**Rust による独立実装の存在そのものが、この目的に資する。**

---

## 9. 提案するロードマップ

| Phase | 内容 | 期間目安 | 成果物 |
|---|---|---|---|
| **P1** | `oxirs-ods`（`ods:` 語彙）実装 + ODP L4 Metadata Exchange 適合の Metadata Server（`oxirs-fuseki` + `oxirs-samm` + `oxirs-stream`）。**L4 適合試験スイートの起草。** | 〜3 か月 | 動作する L4 実装 + 適合試験（OSS, Apache-2.0） |
| **P2** | Discovery Finder / Discovery Service。`oxirs-geosparql` による地理空間 find。`oxirs-samm` ジェネレータによる Discovery Service 生成（ODP 要件 #9）。 | 〜6 か月 | ODS-DFS-L4 の完全実装 |
| **P3** | **`oxirs-prov`（PROV-O）+ VC 署名による検証可能来歴。** §6 の仕様提案とセット。MCP サーバ面（エージェントからの L4 アクセス）。 | 〜9 か月 | DPP / バッテリーパスポート対応の基盤 |
| **P4** | Catena-X（EDC / Dataspace Protocol）との相互接続検証。 | 〜12 か月 | 日 EU 相互運用の実装 |

**P4 について補足**: 2025 年 3 月の Catena-X × ウラノス相互運用 PoC は、「認証方式」「プロトコル」「データモデル」の差分を吸収する**中間層を手作業で構築**することで達成された。この中間層は、構造的には**セマンティック・メディエーション**の問題であり、`oxirs-federate`（SPARQL SERVICE プランナ, 2PC）+ `oxirs-samm` + `oxirs-did` が扱う領域そのものである。

なお **COOLJAPAN OÜ はエストニア法人（EU 域内法人）**であり、日欧双方の枠組みから同一コードベースで検証にあたれる立場にある。

---

## 10. 確認したい事項（DADC / NEDO 各位へ）

1. **L4 の適合試験スイート**は存在するか。無い場合、実装側から起草・提供する用意がある。
2. ODP L4 の **lineage / version 管理の語彙**について、指定または検討中のものはあるか。無い場合、**W3C PROV-O の採用**を提案したい（§6）。
3. **`ods:` 名前空間のオントロジー定義**（Turtle / OWL）は公開されているか。`https://github.com/ODS-DFS-L4/ods/` の実体を確認したい。
4. 「ODS セマンティクスポリシー」（ODP L4 が SHALL で準拠を求めている文書）の所在を確認したい。
5. **Ontology Product（L4 → L2）**のリファレンス実装として、現行の `L4-crawler-service` 以外の想定・ロードマップはあるか。
6. **実装言語の多様性**について、DADC としての方針を伺いたい。Rust による独立実装は、ベンダーロックイン回避および供給網主権の観点で寄与しうると考えている。

---

## 付録 A: OxiRS クレート一覧と ODS-RAM 対応

| クレート | テスト数 | ODS-RAM 対応 |
|---|---|---|
| `oxirs-core` | 2,332 | **L4**（RDF 1.2, 7 形式） |
| `oxirs-arq` | 2,688 | **L4**（SPARQL クエリエンジン） |
| `oxirs-fuseki` | 2,144 | **L4**（SPARQL 1.1/1.2 HTTP, Metadata Endpoint）/ **L3**（ReBAC, OAuth2/OIDC/SAML） |
| `oxirs-samm` | 1,409 | **L4**（SAMM 2.0–2.3 + AAS, 16 ジェネレータ）※Jena 非対応 |
| `oxirs-shacl` | 2,008 | **L1**（データ品質・整合性検証, W3C 27/27, SHACL-AF） |
| `oxirs-shacl-ai` | 1,589 | **L1**（シェイプ推論・データ修復提案） |
| `oxirs-did` | 1,043 | **L3**（W3C DID/VC, 署名付き RDF RDFC-1.0, Ed25519, トラストチェーン） |
| `oxirs-stream` | 1,505 | **L4**（RDF Patch / SPARQL Update デルタ, Kafka/NATS） |
| `oxirs-federate` | 1,397 | **L4**（SPARQL SERVICE プランナ, 2PC, フェデレーション認証） |
| `oxirs-geosparql` | 1,713 | **L4**（GeoSPARQL OGC 1.1 — Discovery 地理空間検索） |
| `oxirs-star` | 1,628 | **L4**（RDF-star / SPARQL-star — 来歴付与の基盤） |
| `oxirs-rule` | 2,072 | **L4**（RDFS / OWL 2 DL / SWRL 推論） |
| `oxirs-tdb` | 2,005 | **L4**（Metadata Store, TDB2 互換・六索引） |
| `oxirs-cluster` | 1,489 | 非機能（Raft 分散） |
| `oxirs-gql` | 2,081 | **L2**（GraphQL ファサード） |
| `oxirs-wasm` | 858 | **L4**（ブラウザ/エッジ Metadata Client） |
| `oxirs-vec` / `oxirs-embed` | 1,598 / 1,345 | **L4**（ベクトル検索・KG 埋め込み — Discovery キーワード検索） |
| `oxirs-ttl` | 1,726 | **L4**（Turtle / TriG） |
| `oxirs-tsdb` / `oxirs-modbus` / `oxirs-canbus` / `oxirs-physics` | 4,410 | 産業 IoT / デジタルツイン |
| `oxirs-graphrag` / `oxirs-chat` | 2,130 | AI（L4 の上位活用） |
| `oxirs (CLI)` | 1,615 | 運用ツール |
| **合計** | **約 43,500** | |

**ライセンス**: Apache-2.0
**リポジトリ**: https://github.com/cool-japan/oxirs
**crates.io**: 全 26 クレート公開済

---

## 付録 B: 参照した仕様・実装

- Open Data Spaces Reference Architecture Model (ODS-RAM) V2 — https://open-dataspaces.gitbook.io/ods-docs/
- Open Data Spaces Protocols (ODP) V1 — 同上
- Open Data Spaces 公開 OSS（IPA 公式 GitHub）— https://github.com/open-dataspaces
- Why Open Dataspaces: 設計思想とアーキテクチャパラダイム（IPA DADC, 2026-04-01）
- ウラノス・エコシステムの実現のためのデータ連携システム構築・実証事業（NEDO, P25008）

---

*本レポートは Apache-2.0 の下で公開し、OxiRS リポジトリの `docs/ODS-RAM-CONFORMANCE.md` としても配置する予定である。誤り・認識違いの指摘を歓迎する。*
