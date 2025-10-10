# RAG検索機能の統合実装計画

**日付**: 2025-10-10
**作業内容**: `generate_note.py` に RAG検索機能を統合

## 概要

現在、`generate_rag_multimodal.py` によってベクトルDBへのデータ保存は完了しているが、`generate_note.py` の分析・要約生成時にRAG検索を活用していない。この計画では、以下の2つの関数にRAG検索を統合する：

1. `analyze_file_with_gemini()` - 個別ファイル分析時に関連情報を取得
2. `generate_final_reflection_note()` - 全体振り返り生成時にプロジェクト全体の文脈を取得

## 専門家からのアドバイス

### Gemini MCP（コード実装）
- **プロンプト構造**: 検索結果（コンテキスト）、指示、質問を明確に分離（XMLタグ推奨）
- **top_k**: 3-5件が開始点として適切
- **出典明記**: 各チャンクにファイル名・ページ等の出典情報を含める
- **フォーマット**: XMLタグで構造化（`<document><source>...</source><content>...</content></document>`）

### o3 MCP（アーキテクチャ設計）
- **マルチクエリ戦略**: ファイル名＋内容要約＋識別子を組み合わせたクエリ
- **階層型RAG**: サマリインデックス→必要時に原文チャンク取得
- **コンテキスト予算配分**:
  - 対象ファイル原文: 60%
  - 近傍/同モジュール: 25%
  - 外部/全体コンテキスト: 15%
- **エラーハンドリング**: RAG失敗時は対象ファイルのみで処理続行

## 実装方針

### 1. RAGSearcher の初期化

`generate_note.py` の `main()` 関数でRAGSearcherを初期化し、各関数に渡す。

```python
from rag.rag_search import RAGSearcher

rag_searcher = RAGSearcher(
    vector_bucket_name=os.getenv('VECTOR_BUCKET_NAME', 'lisa-poc-vectors'),
    index_name=os.getenv('VECTOR_INDEX_NAME', 'project-documents'),
    aws_region=os.getenv('AWS_REGION', 'us-west-2')
)
```

### 2. `analyze_file_with_gemini()` の修正

#### 現在の実装（lines 489-563）
- 対象ファイルのテキストをそのままGemini APIに送信
- 関連情報は取得していない

#### 変更内容
1. **RAG検索の追加** (関数の最初の部分)
   - クエリ: `f"{file_name}に関連する情報"`（シンプルな日本語クエリ）
   - top_k: 3件（コンテキストウィンドウを節約）
   - フィルタ: 同じproject_name

2. **プロンプトへのコンテキスト追加**
   - XMLタグで構造化: `<参考資料>...</参考資料>`
   - 各チャンクを `<document>` タグで囲む
   - 出典（ファイル名）を明記

3. **エラーハンドリング**
   - RAG検索失敗時は警告ログを出力して処理続行
   - 検索結果が0件の場合もエラーにせず続行

#### 実装イメージ
```python
def analyze_file_with_gemini(
    file_name: str,
    file_text: str,
    project_name: str,
    rag_searcher: RAGSearcher  # 追加
) -> Optional[str]:
    # RAG検索で関連情報を取得
    rag_context = ""
    try:
        search_results = rag_searcher.search_by_project(
            project_name=project_name,
            query=f"{file_name}に関連する情報",
            top_k=3
        )

        if search_results:
            rag_context = "<参考資料>\n"
            for result in search_results:
                metadata = result.get('metadata', {})
                rag_context += f"<document>\n"
                rag_context += f"  <source>{metadata.get('file_name', '不明')}</source>\n"
                rag_context += f"  <content>\n{result.get('text', '')}\n</content>\n"
                rag_context += f"</document>\n"
            rag_context += "</参考資料>\n\n"
    except Exception as e:
        print(f"[WARN] RAG検索に失敗しましたが処理を続行します: {e}")

    # プロンプト構築時にRAGコンテキストを追加
    prompt = f"""
{rag_context}
あなたは優秀なビジネスアナリストです。
以下のファイルの内容を分析し、重要なポイントをまとめてください。

【ファイル名】: {file_name}

【内容】:
{file_text}

...（以下既存のプロンプト）
"""
```

### 3. `generate_final_reflection_note()` の修正

#### 現在の実装（lines 575-614）
- 各ファイルの分析結果を結合してGemini APIに送信
- プロジェクト全体の文脈は取得していない

#### 変更内容
1. **RAG検索の追加**
   - クエリ: `f"{project_name}プロジェクトの概要と主要トピック"`
   - top_k: 5件（全体振り返りなのでやや多め）
   - フィルタ: 同じproject_name

2. **プロンプトへのコンテキスト追加**
   - analyze_file_with_gemini()と同様にXML形式で構造化
   - 「プロジェクト全体の参考情報」として明示

3. **コンテキスト予算の配分**
   - 各ファイルの分析結果（既存）: 優先度高
   - RAG検索結果: 補足情報として追加

#### 実装イメージ
```python
def generate_final_reflection_note(
    all_results: List[Dict[str, str]],
    project_name: str,
    rag_searcher: RAGSearcher  # 追加
) -> str:
    # RAG検索でプロジェクト全体の文脈を取得
    rag_context = ""
    try:
        search_results = rag_searcher.search_by_project(
            project_name=project_name,
            query=f"{project_name}プロジェクトの概要と主要トピック",
            top_k=5
        )

        if search_results:
            rag_context = "\n## プロジェクト全体の参考情報\n\n<参考資料>\n"
            for result in search_results:
                metadata = result.get('metadata', {})
                rag_context += f"<document>\n"
                rag_context += f"  <source>{metadata.get('file_name', '不明')}</source>\n"
                rag_context += f"  <content>\n{result.get('text', '')}\n</content>\n"
                rag_context += f"</document>\n"
            rag_context += "</参考資料>\n"
    except Exception as e:
        print(f"[WARN] RAG検索に失敗しましたが処理を続行します: {e}")

    # プロンプト構築（既存の分析結果 + RAGコンテキスト）
    prompt = f"""
{rag_context}

あなたは優秀なプロジェクトマネージャーです。
以下の各ファイルの分析結果をもとに、プロジェクト全体の振り返りノートを作成してください。

## 各ファイルの分析結果
{combined_results}

...（以下既存のプロンプト）
"""
```

### 4. `main()` 関数の修正

RAGSearcherを初期化し、各関数に渡すように変更。

```python
def main(
    input_dir: str,
    output_file: str,
    project_name: str,
    excluded_files: Optional[List[str]] = None,
):
    # RAGSearcher初期化
    try:
        rag_searcher = RAGSearcher(
            vector_bucket_name=os.getenv('VECTOR_BUCKET_NAME', 'lisa-poc-vectors'),
            index_name=os.getenv('VECTOR_INDEX_NAME', 'project-documents'),
            aws_region=os.getenv('AWS_REGION', 'us-west-2')
        )
        print("[INFO] RAGSearcher initialized successfully")
    except Exception as e:
        print(f"[WARN] RAGSearcher initialization failed: {e}")
        print("[INFO] Proceeding without RAG search functionality")
        rag_searcher = None

    # 各ファイルを分析
    for file_name, file_text in files_to_analyze:
        result = analyze_file_with_gemini(
            file_name,
            file_text,
            project_name,
            rag_searcher  # 追加
        )

    # 最終的な振り返りノートを生成
    final_note = generate_final_reflection_note(
        all_results,
        project_name,
        rag_searcher  # 追加
    )
```

## 変更ファイル

- `/Users/kobayashi.masahiro/CMProject/prj-da-2030/lisa/poc/generate_note.py`

## 変更内容サマリ

1. **import追加**: `from rag.rag_search import RAGSearcher`
2. **main()**: RAGSearcherの初期化と各関数への受け渡し
3. **analyze_file_with_gemini()**:
   - 引数追加: `rag_searcher: Optional[RAGSearcher]`
   - RAG検索実行（top_k=3）
   - プロンプトへのコンテキスト追加（XML形式）
4. **generate_final_reflection_note()**:
   - 引数追加: `rag_searcher: Optional[RAGSearcher]`
   - RAG検索実行（top_k=5）
   - プロンプトへのコンテキスト追加（XML形式）

## エラーハンドリング

- RAGSearcher初期化失敗: 警告を出力して `rag_searcher=None` で続行
- RAG検索失敗: try-exceptでキャッチし、警告ログを出力して処理続行
- 検索結果0件: エラーにせず、RAGコンテキストなしで処理

## テスト方法

1. 既存のベクトルDBにデータがある状態で実行
2. `generate_note.py` を実行して出力を確認
3. 生成されたノートにRAG検索結果が含まれているか確認
4. RAG検索失敗時（ネットワークエラー等）も正常に処理が続行されるか確認

## 期待される効果

- ファイル分析時に関連ファイルの情報も参照できるようになり、より文脈を理解した分析が可能に
- プロジェクト全体の振り返り生成時に、個別ファイルの分析では拾いきれなかった横断的な情報も活用できる
- ベクトルDBに保存された過去のプロジェクト情報も活用できる（将来的に）

## 注意事項

- RAG検索はあくまで補助的な情報源として位置づけ、対象ファイルの内容を最優先
- コンテキストウィンドウを消費しすぎないよう、top_kは控えめに設定
- RAG検索失敗時も処理が止まらないよう、エラーハンドリングを徹底

---

## Codexレビュー結果と段階的実装計画

### レビュー結果サマリ

**設計の妥当性**: 全体として妥当。以下の改善を推奨：
- クエリ生成強化（単一→複数クエリ＋MMR集約）
- XMLプロンプトの明確化（CDATA、doc_id/source/score付き）
- 検索パイプライン分離（一次取得30件→MMR/再ランクで3/5件）

**パフォーマンス**:
- top_k=3/5は最終注入件数として適切
- 一次取得は多め（30件）→MMR/再ランクで圧縮を推奨
- pgvectorインデックス（IVFFLAT/HNSW）のチューニング必要

**保守性**:
- RAGSearcherの責務分離（検索API vs 再ランク/整形）
- 設定の型付き管理（dataclass/pydantic）
- XMLプロンプト生成ヘルパーのユーティリティ化

**落とし穴**:
- クエリの曖昧性（ファイル名が一般名詞の場合）
- 同名ファイルの衝突（project + file_path でフィルタ必須）
- 日本語特性（NFKC正規化、記号・全半角対応）
- XML破損（スニペットのエスケープ/CDATA必須）
- トークン超過（チャンク圧縮要約の実装）

### 段階的実装計画

#### Phase 1: 基本的なRAG統合（今回実装）
**目的**: 最小限の変更でRAG検索を統合し、動作を確認

**実装内容**:
- RAGSearcherの初期化と各関数への受け渡し
- analyze_file_with_gemini(): 単一クエリでRAG検索（top_k=3）
- generate_final_reflection_note(): 単一クエリでRAG検索（top_k=5）
- 基本的なXML形式でプロンプト構築
- エラーハンドリング（初期化失敗・検索失敗時の継続）

**スコープ外**（Phase 2/3で対応）:
- マルチクエリ・MMR・再ランク
- CDATA化・doc_id/score付きXML
- チャンク圧縮要約
- 構造化ログ・タイムアウト制御

**理由**:
- まず動作確認を優先
- 複雑な機能は段階的に追加
- 影響範囲を最小化

#### Phase 2: クエリ強化とMMR導入（将来実装）
**実装内容**:
- マルチクエリ生成（ファイル名、見出し、キーワード）
- MMRによる多様性確保
- 一次取得30件→最終3/5件への圧縮
- 日本語正規化（NFKC）とキーワード抽出

**必要なユーティリティ**:
- `make_queries_for_file()`: ファイルから3-5個のクエリ生成
- `make_queries_for_project()`: プロジェクトから5-7個のクエリ生成
- `apply_mmr()`: MMRアルゴリズム実装

#### Phase 3: XML改善と再ランク（将来実装）
**実装内容**:
- XMLプロンプトのCDATA化
- doc_id/source/score付きの構造化
- 再ランキングモデル導入
- チャンク圧縮要約（200-300トークン）
- 構造化ログとタイムアウト制御

**必要なユーティリティ**:
- `build_xml_context()`: CDATA化・エスケープ処理
- `compress_chunk()`: チャンク要約
- `rerank_results()`: 再ランキング

### Phase 1 実装の具体的変更点

以下、当初計画からの変更なし。Codexの指摘を踏まえた改善はPhase 2/3で対応。

**変更箇所**:
1. `generate_note.py` にRAGSearcherのimport追加
2. `main()` でRAGSearcherを初期化（失敗時はNoneで続行）
3. `analyze_file_with_gemini()` にrag_searcher引数追加、RAG検索実行
4. `generate_final_reflection_note()` にrag_searcher引数追加、RAG検索実行

**エラーハンドリング**:
- RAGSearcher初期化失敗 → 警告 + None継続
- RAG検索失敗 → try-except + 警告 + 処理継続
- 検索結果0件 → エラーにせず継続
