# RAG技術手法ドキュメント

## 概要
LISA PoCで実装されているRAG（Retrieval-Augmented Generation）システムの技術手法を体系的にまとめたドキュメントです。データの取得・処理からベクトル化、検索、生成までの全プロセスで使用している技術を詳細に解説します。

## 1. データ収集・前処理フェーズ

### 1.1 データソース統合
```python
# project_config.yaml による複数データソース管理
projects:
  プロジェクトA:
    google_drive:
      - "フォルダID1"
      - "フォルダID2"
```

**技術的特徴**:
- **マルチソース対応**: Google Drive、Slack（予定）、Backlog（予定）
- **設定駆動型**: YAML設定による柔軟なプロジェクト管理
- **並列取得**: 複数フォルダからの同時データ取得

### 1.2 ファイル形式処理（Unstructured Library）

**対応形式と処理方法**:
```python
# PDFの処理例
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename=file_path,
    strategy="hi_res",  # 高精度OCR有効
    languages=["jpn"],
    max_partition_length=2000
)
```

**処理戦略**:
- **PDF**: `hi_res`（OCR付き高精度）/ `fast`（テキストのみ）
- **Word/Excel/PowerPoint**: python-docx, openpyxl, python-pptx
- **Google Docs**: 自動変換後処理
- **Markdown**: 構造保持パース

### 1.3 テキスト正規化・クリーニング

```python
def clean_text(text: str) -> str:
    # 制御文字除去
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 重複空白の正規化
    text = re.sub(r'\s+', ' ', text)
    # NFKC正規化（全角/半角統一）
    text = unicodedata.normalize('NFKC', text)
    return text.strip()
```

## 2. ベクトル化フェーズ

### 2.1 埋め込み生成（Gemini Embeddings）

```python
# Gemini埋め込みモデル使用
embeddings = GeminiEmbeddings(
    api_key=api_key,
    model_name="gemini-embedding-001",  # 1536次元
    dimension=1536  # アップサンプリング
)

# バッチ処理最適化
def generate_embeddings_batch(texts: List[str], batch_size: int = 20):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

**最適化技術**:
- **バッチ処理**: 20〜100テキスト/バッチ
- **並列化**: ThreadPoolExecutorによる並行処理
- **キャッシング**: 処理済み埋め込みの再利用

### 2.2 チャンク分割戦略

```python
def smart_chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200):
    """意味的境界を考慮したチャンク分割"""
    # 段落・セクション境界での分割優先
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
```

### 2.3 メタデータ付与

```python
metadata = {
    "project_name": project_name,
    "file_name": file_name,
    "doc_type": doc_type,  # "議事録", "提案書", "設計書"等
    "created_at": datetime.now().isoformat(),
    "chunk_id": chunk_id,
    "total_chunks": total_chunks
}
```

## 3. ベクトルストレージフェーズ

### 3.1 AWS S3 Vectors設定

```python
vector_store = S3VectorStore(
    vector_bucket_name="lisa-poc-vectors",
    index_name="project-documents",
    dimension=1536,
    distance_metric="cosine",
    region_name="us-west-2"
)

# インデックス作成（初回のみ）
vector_store.create_index(
    index_type="HNSW",
    index_params={
        "m": 48,
        "ef_construction": 512,
        "ef_search": 512
    }
)
```

**インデックス特性**:
- **HNSW（Hierarchical Navigable Small World）**: 高速近似最近傍探索
- **パラメータチューニング**:
  - `m`: グラフの次数（精度と速度のバランス）
  - `ef_construction`: 構築時の探索幅
  - `ef_search`: 検索時の探索幅

## 4. 検索フェーズ

### 4.1 2段階検索アルゴリズム

```python
def two_stage_search(project_name: str, k: int = 30):
    """2段階検索による精度向上"""

    # Phase 1: プロジェクト名から拡張キーワード生成
    keywords = generate_project_keywords(project_name)
    # 例: "EC売上改善" → "EC 電子商取引 売上 分析 機械学習"

    # Phase 2-1: 現在のプロジェクトから検索
    current_results = retriever.search_similar_documents(
        query=keywords,
        project_name=project_name,
        k=k//2
    )

    # Phase 2-2: プロジェクト概要生成
    project_summary = generate_project_summary(current_results)

    # Phase 2-3: 類似プロジェクトから検索
    similar_query = generate_similar_project_query(project_summary)
    similar_results = retriever.get_cross_project_insights(
        query=similar_query,
        exclude_project=project_name,
        k=k//2
    )

    return current_results + similar_results
```

### 4.2 RAG-Fusion（複数クエリ並行検索）

```python
def rag_fusion_search(base_query: str, num_queries: int = 3):
    """RAG-Fusionによる検索カバレッジ向上"""

    # 1. 複数の異なるクエリを生成
    queries = generate_multiple_queries(base_query, num_queries)
    # 例: ["EC 売上向上 施策", "オンラインストア 収益改善", "電子商取引 KPI分析"]

    # 2. 各クエリで並行検索
    all_results = []
    for query in queries:
        results = vector_store.search(query, k=k*2)
        all_results.append(results)

    # 3. RRF（Reciprocal Rank Fusion）で統合
    return reciprocal_rank_fusion(all_results, k=60)
```

### 4.3 Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(results_list: List[List[Tuple]], k: int = 60):
    """複数の検索結果をRRFで統合"""
    rrf_scores = {}

    for query_idx, results in enumerate(results_list):
        for rank, (doc, distance) in enumerate(results, start=1):
            doc_id = doc.key
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'doc': doc,
                    'rrf_score': 0.0
                }
            # RRFスコア計算: 1/(k+rank)
            rrf_scores[doc_id]['rrf_score'] += 1.0 / (k + rank)

    # RRFスコアでソート
    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )
    return ranked_results
```

### 4.4 ハイブリッドスコアリング

```python
def hybrid_scoring(results: List[Tuple], time_weight: float = 0.2):
    """類似度と時系列の重み付け統合"""
    scored_results = []

    for doc, distance in results:
        # コサイン類似度（0-1に正規化）
        similarity = 1.0 - distance

        # 時系列スコア（指数減衰）
        if 'created_at' in doc.metadata:
            days_old = (datetime.now() - doc.metadata['created_at']).days
            time_score = math.exp(-days_old / 90)  # 90日で半減
        else:
            time_score = 0.5

        # ハイブリッドスコア
        final_score = (1 - time_weight) * similarity + time_weight * time_score
        scored_results.append((doc, final_score))

    return sorted(scored_results, key=lambda x: x[1], reverse=True)
```

## 5. リフレクション再生成フェーズ

### 5.1 初回ノート分析

```python
def analyze_reflection_note(note: str) -> Dict[str, str]:
    """生成されたノートから具体的な情報を抽出"""

    prompt = f"""
    以下のリフレクションノートから抽出してください：
    1. 具体的な技術要素
    2. 業界特有の用語
    3. 主要な課題
    4. 成功要因
    5. 特徴的なキーワード

    ノート: {note}
    """

    response = llm.invoke(prompt)

    return {
        "refined_keywords": extract_keywords(response),
        "technical_stack": extract_tech(response),
        "domain_context": extract_domain(response),
        "key_challenges": extract_challenges(response),
        "success_factors": extract_success(response)
    }
```

### 5.2 改善された再検索

```python
def perform_refined_search(analysis_result: Dict[str, str]):
    """分析結果を使用した精度の高い再検索"""

    # 1. 改善されたクエリ構築
    refined_query = f"""
    {analysis_result['refined_keywords']}
    {analysis_result['technical_stack']}
    {analysis_result['domain_context']}
    """

    # 2. より大きなk値で再検索
    results = vector_store.search(
        query=refined_query,
        k=40,  # 初回の2倍
        min_score=0.3  # より緩い閾値
    )

    # 3. 結果のフィルタリング
    filtered = filter_by_relevance(results, analysis_result)

    return filtered
```

### 5.3 最終ノート統合生成

```python
def regenerate_reflection_note(
    initial_note: str,
    refined_context: str,
    analysis_result: Dict[str, str]
):
    """初回ノートと改善された情報を統合して再生成"""

    prompt = f"""
    【重要】これは2回目の生成です。
    初回生成の内容と、改善された検索結果を統合して、
    より精度の高いリフレクションノートを作成してください。

    初回ノート:
    {initial_note[:3000]}

    プロジェクト分析結果:
    - 技術スタック: {analysis_result['technical_stack']}
    - ドメイン: {analysis_result['domain_context']}
    - 主要課題: {analysis_result['key_challenges']}
    - 成功要因: {analysis_result['success_factors']}

    改善された検索結果:
    {refined_context}

    統合的で洞察に富んだリフレクションノートを生成してください。
    """

    return llm.invoke(prompt)
```

## 6. パフォーマンス最適化技術

### 6.1 バッチ処理

```python
# 埋め込み生成の並列バッチ処理
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for batch in batches:
        future = executor.submit(generate_embeddings, batch)
        futures.append(future)

    results = [f.result() for f in futures]
```

### 6.2 キャッシング戦略

```python
class EmbeddingCache:
    def __init__(self):
        self.cache = {}
        self.cache_file = "embeddings_cache.pkl"

    def get_or_generate(self, text: str):
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.cache:
            return self.cache[text_hash]

        embedding = generate_embedding(text)
        self.cache[text_hash] = embedding
        self.save_cache()

        return embedding
```

### 6.3 インクリメンタル処理

```python
def incremental_rag_update(new_documents: List[Document]):
    """新規ドキュメントのみを処理"""

    # 既存のドキュメントIDを取得
    existing_ids = vector_store.get_all_document_ids()

    # 新規ドキュメントのみフィルタ
    new_docs = [
        doc for doc in new_documents
        if doc.id not in existing_ids
    ]

    # 新規分のみ処理
    if new_docs:
        embeddings = generate_embeddings_batch(new_docs)
        vector_store.add_documents(new_docs, embeddings)
```

## 7. エラーハンドリング・リトライ戦略

### 7.1 指数バックオフリトライ

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def call_gemini_api(prompt: str):
    """Gemini APIの呼び出しとリトライ処理"""
    try:
        response = gemini_client.generate_content(prompt)
        return response
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            raise GeminiQuotaError(str(e))
        raise
```

### 7.2 フォールバック戦略

```python
def search_with_fallback(query: str):
    """段階的フォールバック検索"""
    try:
        # 1. Enhanced RAGを試行
        if USE_ENHANCED_RAG:
            return enhanced_rag_search(query)
    except Exception as e:
        log.warning(f"Enhanced RAG failed: {e}")

    try:
        # 2. RAG-Fusionを試行
        if USE_RAG_FUSION:
            return rag_fusion_search(query)
    except Exception as e:
        log.warning(f"RAG-Fusion failed: {e}")

    # 3. 標準検索にフォールバック
    return standard_search(query)
```

## 8. Self-RAGとの関係（未実装）

### 8.1 Self-RAGの概念

Self-RAG（Self-Reflective RAG）は、生成プロセス中に自己反省トークンを使用して動的に品質を評価・制御する手法です：

```python
# Self-RAGの概念（未実装）
class SelfRAG:
    """Self-RAGの概念的な実装"""

    def generate_with_reflection(self, query: str):
        # 1. 検索の必要性判断（ISREL トークン）
        needs_retrieval = self.assess_retrieval_need(query)

        if needs_retrieval:
            docs = self.retrieve(query)

            # 2. 各ドキュメントの関連性評価
            for doc in docs:
                relevance = self.assess_relevance(doc, query)  # ISREL
                if relevance > threshold:
                    # 3. 生成
                    output = self.generate_segment(doc)

                    # 4. サポート性評価（ISSUP トークン）
                    is_supported = self.assess_support(output, doc)

                    # 5. 有用性評価（ISUSE トークン）
                    is_useful = self.assess_usefulness(output, query)

                    if is_supported and is_useful:
                        return output
```

### 8.2 現在の実装との比較

| 特徴 | Self-RAG（未実装） | 現在の実装 |
|------|------------------|-----------|
| **反省メカニズム** | リアルタイム反省トークン | 生成後の分析（2パス） |
| **検索判断** | 動的（必要時のみ） | 常に実行 |
| **品質評価** | 生成中に各セグメントを評価 | 生成後に全体を評価 |
| **再生成** | セグメント単位で即座に | ノート全体を再生成 |
| **実装複雑度** | 高（カスタムモデル必要） | 中（API呼び出しのみ） |

### 8.3 現在の「再生成機能」が取り入れているSelf-RAG要素

現在の実装は完全なSelf-RAGではありませんが、以下の要素を部分的に取り入れています：

```python
# 現在の実装：2パス方式の自己改善
def generate_with_refinement():
    # Phase 1: 初回生成
    initial_note = generate_initial()

    # Phase 2: 自己分析（Self-RAG的要素）
    analysis = analyze_reflection_note(initial_note)

    # Phase 3: 改善検索
    refined_context = perform_refined_search(analysis)

    # Phase 4: 再生成
    final_note = regenerate_with_improvements(
        initial_note,
        refined_context,
        analysis
    )

    return final_note
```

**Self-RAG的な要素**:
- ✅ 生成結果の品質評価
- ✅ 改善のための再検索
- ✅ より良い結果の生成
- ❌ リアルタイム反省トークン
- ❌ セグメント単位の評価
- ❌ 動的な検索判断

### 8.4 将来の実装計画

完全なSelf-RAGの実装には以下が必要です：

1. **カスタムLLMファインチューニング**: 反省トークンの生成能力
2. **ストリーミング生成**: リアルタイムの評価と制御
3. **セグメント管理**: 細粒度の生成と評価
4. **動的検索制御**: 必要性に応じた検索の有効/無効

現状では、2パス方式の再生成機能でSelf-RAGの主要な利点（自己改善）を実現しています。

## まとめ

LISA PoCのRAGシステムは、以下の主要技術を組み合わせて高精度な検索と生成を実現しています：

1. **データ処理**: Unstructured Libraryによる多様なファイル形式対応
2. **ベクトル化**: Gemini Embeddingsとスマートチャンキング
3. **ストレージ**: AWS S3 VectorsのHNSWインデックス
4. **検索**: 2段階検索、RAG-Fusion、RRF統合
5. **再生成**: ノート分析と改善検索による精度向上（Self-RAG的要素）
6. **最適化**: バッチ処理、キャッシング、インクリメンタル更新

これらの技術により、社内ドキュメントから高品質な知見抽出とインサイト生成を実現しています。

**実装済みのRAG拡張**:
- ✅ RAG-Fusion（複数クエリ並行検索）
- ✅ Enhanced RAG（拡張検索機能）
- ✅ 再生成機能（Self-RAG的要素を部分的に実装）

**未実装の技術**:
- ❌ 完全なSelf-RAG（リアルタイム反省トークン）
- ❌ GraphRAG（グラフベースの知識表現）
- ❌ Agentic RAG（エージェント型の自律的検索）
