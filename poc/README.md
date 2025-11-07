# LISA PoC - AI駆動型プロジェクト分析システム

プロジェクトのドキュメントをRAG (Retrieval-Augmented Generation) とGoogle Gemini APIを使用して分析し、インサイトやリフレクションノートを自動生成するシステムです。

## 🎯 主要機能

### 1. 📊 プロジェクト分析とリフレクションノート生成
- RAGを活用した過去プロジェクトとの比較分析
- 成功要因・リスク・改善提案の自動抽出
- 2段階生成プロセスによる高精度なノート作成

### 2. 👤 ペルソナ管理システム
- 生のヒアリングデータから個人ペルソナを自動生成
- 複数の個人ペルソナを統合してチーム全体のペルソナを作成
- ペルソナの完全性チェック機能

### 3. 📝 ドキュメント生成
- ヒアリングシート自動生成（リフレクションノートから）
- 提案書自動生成（ヒアリングシートから）
- RAG検索による過去事例の活用

### 4. 🧠 思考プロセス分析
- LLMの思考プロセスを詳細に記録・分析
- プロンプト改善のためのインサイト抽出
- ドキュメント生成の判断基準を可視化

### 5. 🔍 高度なRAG機能
- AWS S3 Vectorsによる高速ベクトル検索
- 2段階検索アルゴリズム
- RAG-Fusion（複数クエリ並行検索）
- CRAG（Corrective RAG）による適応的検索

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                     データソース層                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐            │
│  │Google Drive│ │   Slack    │ │  Backlog   │            │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘            │
│        │              │ (将来)        │ (将来)             │
└────────┼──────────────┴──────────────┴────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────┐
│                    処理層 (Python)                           │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Unstructured Library (PDF/Word/Excel/PPT解析)    │      │
│  └──────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Google Gemini API (google-genai SDK)             │      │
│  │  - Embedding: gemini-embedding-001 (1536次元)     │      │
│  │  - Generation: gemini-2.5-flash/pro               │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────┐
│                  ベクトルストレージ層                          │
│  ┌──────────────────────────────────────────────────┐      │
│  │        AWS S3 Vectors (プレビュー版)               │      │
│  │  - サーバーレス・低コスト                          │      │
│  │  - サブ秒レスポンス                               │      │
│  │  - 最大50Mベクトル/インデックス                    │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 📁 ディレクトリ構造

```
lisa/poc/
├── generate_note.py                 # リフレクションノート生成
├── generate_document.py             # ドキュメント（ヒアリングシート・提案書）生成
├── generate_rag_unstructured.py     # RAGインデックス構築
├── process_raw_hearing.py           # 個人ペルソナ生成
├── generate_persona.py              # 統合ペルソナ生成
├── check_persona_completeness.py    # ペルソナ品質チェック
├── thought_process_analyzer.py      # リフレクションノート用思考プロセス分析
├── document_thought_analyzer.py     # ドキュメント用思考プロセス分析
├── delete_s3_vectors.py             # S3 Vectors管理
├── project_config.yaml              # プロジェクト設定
├── generators/                      # ドキュメント生成器
│   ├── hearing_sheet_generator.py
│   └── proposal_generator.py
├── rag/                            # RAGシステム
│   ├── rag_retriever.py
│   ├── vector_store.py
│   ├── embeddings.py
│   └── enhanced_rag_search.py
├── outputs/                        # 出力ファイル
├── prompts/                        # プロンプトテンプレート
└── plans/                          # 実装計画書
```

## 🚀 セットアップ

### 1. 依存ライブラリのインストール

#### システム依存関係（macOS）

このプロジェクトでは、PDF処理のために `poppler-utils` が必要です。

```bash
# Homebrew がインストールされていない場合
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# poppler-utils のインストール（pdfinfo, pdftotext等を含む）
brew install poppler

# インストール確認
which pdfinfo
pdfinfo --version
```

#### Python依存関係

```bash
cd lisa/poc

# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

主要な依存ライブラリ:
- `google-genai`: Google Gemini API SDK
- `unstructured[pdf,md,docx,xlsx,pptx]`: ファイル解析（**poppler必須**）
- `langchain-google-genai`: LangChain統合
- `boto3`: AWS S3 Vectors連携
- `func-timeout`: タイムアウト処理

#### オプションの依存関係

```bash
# PyMuPDF（PDF処理の高速化 - 推奨）
pip install PyMuPDF

# tesseract（OCR処理用 - オプション）
brew install tesseract
brew install tesseract-lang  # 追加言語パック

# LibreOffice（Office文書変換用 - オプション）
brew install --cask libreoffice
```

#### インストール確認

```bash
# 基本的な動作確認
python3 -c "from unstructured.partition.pdf import partition_pdf; print('PDF processing ready')"

# pdfinfo の動作確認
pdfinfo --version

# スクリプトの実行
python3 generate_rag_unstructured.py --help
```

### 2. Google Cloud Console設定

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. プロジェクト作成
3. 「APIとサービス」→「ライブラリ」→「Google Drive API」を有効化
4. 「認証情報を作成」→「OAuthクライアントID」→「デスクトップアプリ」
5. `credentials.json` をダウンロード → `lisa/poc/credentials.json` に配置

### 3. 環境変数設定

```bash
cp .env.example .env
vi .env
```

#### 必須の環境変数

```bash
# Google Gemini API Key（必須）
GEMINI_API_KEY=your-gemini-api-key-here
```

取得方法: [Google AI Studio](https://aistudio.google.com/app/apikey)

#### AWS S3 Vectors設定（RAG機能使用時に必須）

```bash
# AWSリージョン
AWS_REGION=us-west-2

# S3ベクトルバケット名とインデックス名
VECTOR_BUCKET_NAME=lisa-poc-vectors
VECTOR_INDEX_NAME=project-documents

# 埋め込みモデル設定
EMBEDDING_MODEL=gemini-embedding-001
DIMENSION=1536
```

#### オプションの環境変数

<details>
<summary>基本設定</summary>

```bash
# Geminiモデル選択（デフォルト: gemini-2.5-flash）
GEMINI_MODEL=gemini-2.5-flash

# AWS認証（特定プロファイル使用時のみ）
# AWS_PROFILE=your-profile-name
```
</details>

<details>
<summary>思考プロセス分析機能（デフォルト: 有効）</summary>

```bash
# 思考プロセス分析を有効化（true/false、デフォルト: true）
ENABLE_THOUGHT_ANALYSIS=true

# 思考プロセス分析用のモデル（デフォルト: GEMINI_MODEL）
THOUGHT_ANALYSIS_MODEL=gemini-2.5-flash
```

詳細: [THOUGHT_ANALYSIS_README.md](THOUGHT_ANALYSIS_README.md)
</details>

<details>
<summary>リフレクションノート再生成機能（デフォルト: 有効）</summary>

```bash
# 再生成を有効化（true/false、デフォルト: true）
ENABLE_REFLECTION_REFINEMENT=true

# 再検索時の取得件数
RAG_REFINEMENT_K_CURRENT=20
RAG_REFINEMENT_K_SIMILAR=20
RAG_REFINEMENT_MAX_TOTAL=40
```
</details>

<details>
<summary>RAG検索チューニング</summary>

```bash
# 基本検索設定
RAG_MIN_SCORE=0.6                  # 最小類似度スコア（デフォルト: 0.6）
RAG_SCORING_METHOD=hybrid           # 評価方法（デフォルト: hybrid）
RAG_TIME_WEIGHT=0.2                 # 時系列重み（デフォルト: 0.2）
RAG_DECAY_DAYS=90                   # 時間減衰半減期（デフォルト: 90）

# RAG専用モード時の設定
RAG_ONLY_MODE_K_CURRENT=15
RAG_ONLY_MODE_K_SIMILAR=15
RAG_ONLY_MODE_MAX_TOTAL=30
RAG_ONLY_MODE_MIN_SCORE=0.3
```
</details>

<details>
<summary>RAG-Fusion設定（デフォルト: 有効）</summary>

```bash
# RAG-Fusion有効化（デフォルト: true）
USE_RAG_FUSION=true

# 生成するクエリ数（デフォルト: 3）
RAG_FUSION_NUM_QUERIES=3
```
</details>

<details>
<summary>CRAG（Corrective RAG）設定（デフォルト: 無効）</summary>

```bash
# CRAG有効化（デフォルト: false）
ENABLE_CRAG=false

# 関連性判定閾値
CRAG_UPPER_THRESHOLD=0.5
CRAG_LOWER_THRESHOLD=-0.5

# Knowledge Refinement設定
USE_KNOWLEDGE_REFINEMENT=true
MAX_REFINED_SEGMENTS=5
```

詳細: [CRAG_IMPLEMENTATION.md](CRAG_IMPLEMENTATION.md)
</details>

<details>
<summary>パフォーマンスチューニング</summary>

```bash
# ファイル処理タイムアウト（秒、デフォルト: 300）
PARTITION_TIMEOUT=300

# PDFの高解像度処理（デフォルト: true）
PDF_USE_HI_RES=true

# バッチサイズ（デフォルト: 20）
BATCH_SIZE=20

# 並列処理数（デフォルト: 4）
MAX_WORKERS=4
```
</details>

### 4. プロジェクト設定ファイルの作成

```bash
cp project_config.yaml.sample project_config.yaml
vi project_config.yaml
```

設定例:
```yaml
projects:
  プロジェクトA:
    google_drive:
      - "フォルダID1"
      - "フォルダID2"
```

## 📖 使用方法

### ワークフロー全体図

```
1. ペルソナ作成
   ├─ process_raw_hearing.py      # 個人ペルソナ生成
   ├─ generate_persona.py          # 統合ペルソナ生成
   └─ check_persona_completeness.py # 品質チェック
         ↓
2. RAGインデックス構築
   └─ generate_rag_unstructured.py
         ↓
3. リフレクションノート生成
   └─ generate_note.py
         ↓
4. ドキュメント生成
   ├─ generate_document.py hearing-sheet  # ヒアリングシート
   └─ generate_document.py proposal       # 提案書
```

### 1. ペルソナ管理

#### 1.1 個人ペルソナの生成

生のヒアリングデータから構造化された個人ペルソナを生成します。

```bash
# 全ての raw_hearing ファイルを処理
python process_raw_hearing.py

# 特定の人のみ処理
python process_raw_hearing.py --person "安達さん"

# 既存ファイルを上書きして再生成
python process_raw_hearing.py --force
```

**入力**: `ドキュメント/raw_hearing/{名前}.md`
**出力**: `outputs/ドキュメント/hearing/{名前}_persona.md`

#### 1.2 統合ペルソナの生成

複数の個人ペルソナを統合して、チーム全体のペルソナを作成します。

```bash
# 全個人ペルソナを統合
python generate_persona.py

# 特定の人のペルソナのみ統合
python generate_persona.py --person "安達さん" "山田さん"

# 既存ファイルを上書き
python generate_persona.py --force
```

**入力**: `outputs/ドキュメント/hearing/*_persona.md`
**出力**: `outputs/specialist_persona_prompt_latest.md`

#### 1.3 ペルソナの品質チェック

生成された統合ペルソナの完全性を確認します。

```bash
# デフォルトファイルをチェック
python check_persona_completeness.py

# 特定のファイルをチェック
python check_persona_completeness.py --file outputs/specialist_persona_prompt_latest.md
```

**チェック内容**:
- 必須セクションの存在確認
- 途中切れの検出（文章が未完成でないか）
- 文字数・行数の妥当性

### 2. RAGインデックスの構築

```bash
# 全プロジェクトのインデックスを構築
python generate_rag_unstructured.py

# 特定のプロジェクトのみ処理
python generate_rag_unstructured.py --project "プロジェクトA"

# キャッシュをクリアして再構築
python generate_rag_unstructured.py --clear-cache

# 詳細ログを表示
python generate_rag_unstructured.py --verbose
```

### 3. リフレクションノートの生成

```bash
# 基本的な使用方法
python generate_note.py

# 特定のプロジェクトのみ処理
python generate_note.py --project "プロジェクトA"

# CRAG機能を有効にして実行
python generate_note.py --enable-crag

# 思考プロセス分析を有効にして実行
python generate_note.py --enable-thought-analysis
```

**出力**:
- `outputs/{project_name}/reflection_note_latest.md` - リフレクションノート
- `outputs/{project_name}/analysis/thought_process_latest.json` - 思考プロセス（オプション）

#### リフレクションノート生成の仕組み

**Phase 1: 初回生成**
1. プロジェクト名から拡張キーワードを生成
2. RAG検索で過去プロジェクト情報を取得
3. 初回リフレクションノートを生成

**Phase 2: 分析と再生成**（`ENABLE_REFLECTION_REFINEMENT=true`の場合）
1. 初回ノートを分析し、技術要素・課題・成功要因を抽出
2. 抽出した情報で改善された検索クエリを生成
3. より関連性の高い過去プロジェクト情報を再検索
4. 初回ノートと新しい情報を統合して再生成

**Phase 3: 思考プロセス分析**（`ENABLE_THOUGHT_ANALYSIS=true`の場合）
1. 生成されたノートの思考プロセスを分析
2. 判断基準、情報源、推論ロジックを記録
3. JSON形式で保存

### 4. ドキュメント生成

#### 4.1 ヒアリングシート生成

```bash
# リフレクションノートからヒアリングシート生成
python generate_document.py hearing-sheet --input outputs/project/reflection_note_latest.md

# CRAG機能を有効にして生成
python generate_document.py hearing-sheet --input reflection_note.md --enable-crag

# 追加の指示を与えて生成
python generate_document.py hearing-sheet --input reflection_note.md \
    --additional-prompt "期限が厳しいので重要事項に絞る"

# 思考プロセス分析を有効にして生成
python generate_document.py hearing-sheet --input reflection_note.md --enable-thought-analysis
```

**出力**:
- `{input_dir}/{timestamp}_hearing_sheet.md` - ヒアリングシート
- `{input_dir}/hearing_sheet_latest.md` - 最新版
- `{input_dir}/analysis/hearing_sheet_thought_process_latest.json` - 思考プロセス（オプション）

#### 4.2 提案書生成

```bash
# ヒアリングシートから提案書生成
python generate_document.py proposal --input outputs/project/hearing_sheet_latest.md

# リフレクションノートから直接生成も可能
python generate_document.py proposal --input outputs/project/reflection_note_latest.md

# プロジェクト情報を指定
python generate_document.py proposal --input hearing_sheet.md \
    --project-name "ECサイトリニューアル" \
    --customer-name "株式会社サンプル"
```

**出力**:
- `{input_dir}/{timestamp}_proposal.md` - 提案書
- `{input_dir}/proposal_latest.md` - 最新版
- `{input_dir}/analysis/proposal_thought_process_latest.json` - 思考プロセス（オプション）

### 5. S3 Vectorsの管理

```bash
# インデックスのみ削除
python delete_s3_vectors.py --index-only

# バケットとインデックスを削除
python delete_s3_vectors.py

# 特定のバケット/インデックスを削除
python delete_s3_vectors.py --bucket my-bucket --index my-index

# Dry-runモード（削除せず確認のみ）
python delete_s3_vectors.py --dry-run
```

## 📊 出力ファイル構造

```
outputs/
├── specialist_persona_prompt_latest.md      # 統合ペルソナ
├── ドキュメント/
│   └── hearing/
│       ├── 安達さん_persona.md              # 個人ペルソナ
│       └── 山田さん_persona.md
├── プロジェクトA/
│   ├── reflection_note_latest.md           # リフレクションノート
│   ├── {timestamp}_reflection_note.md
│   ├── hearing_sheet_latest.md             # ヒアリングシート
│   ├── {timestamp}_hearing_sheet.md
│   ├── proposal_latest.md                  # 提案書
│   ├── {timestamp}_proposal.md
│   └── analysis/                           # 思考プロセス分析結果
│       ├── thought_process_latest.json
│       ├── hearing_sheet_thought_process_latest.json
│       └── proposal_thought_process_latest.json
└── プロジェクトB/
    └── ...
```

## 🔧 トラブルシューティング

### 基本的なエラー

<details>
<summary>credentials.json が見つかりません</summary>

Google Cloud Consoleから`credentials.json`をダウンロードし、`lisa/poc/`に配置してください。
</details>

<details>
<summary>GEMINI_API_KEY が設定されていません</summary>

`.env`ファイルに`GEMINI_API_KEY`を設定してください。
APIキーは[Google AI Studio](https://aistudio.google.com/app/apikey)から取得できます。
</details>

<details>
<summary>project_config.yaml が見つかりません</summary>

```bash
cp project_config.yaml.sample project_config.yaml
vi project_config.yaml
```
</details>

### PDF処理関連

<details>
<summary>FileNotFoundError: [Errno 2] No such file or directory: 'pdfinfo'</summary>

**原因**: `poppler-utils` がインストールされていません。

**解決方法**:
```bash
brew install poppler

# インストール確認
which pdfinfo
pdfinfo --version
```
</details>

<details>
<summary>[WARN] PyMuPDF not installed. PDF text detection disabled.</summary>

**影響**: 基本的なPDF処理は動作しますが、テキスト検出が遅くなる可能性があります。

**解決方法（オプション）**:
```bash
pip install PyMuPDF

# インストール確認
python3 -c "import fitz; print('PyMuPDF installed')"
```
</details>

### AWS S3 Vectors関連

<details>
<summary>S3 Vectorsのアクセスエラー</summary>

```bash
# AWS CLIの設定を確認
aws configure list

# 特定のプロファイルを使用
export AWS_PROFILE=your-profile-name

# リージョンを確認
export AWS_REGION=us-west-2
```
</details>

<details>
<summary>ベクトル検索が機能しない</summary>

```bash
# インデックスの再構築
python generate_rag_unstructured.py --clear-cache

# S3 Vectorsのステータス確認
aws s3api head-bucket --bucket $VECTOR_BUCKET_NAME
```
</details>

### パフォーマンス関連

<details>
<summary>ファイルの処理が終わらない</summary>

```bash
# .envファイルでタイムアウト設定を調整
PARTITION_TIMEOUT=600

# PDFの処理戦略を変更
PDF_USE_HI_RES=false
```
</details>

<details>
<summary>メモリ不足エラー</summary>

```bash
# バッチサイズを調整
BATCH_SIZE=10

# 並列処理数を制限
MAX_WORKERS=2
```
</details>

### ペルソナ生成関連

<details>
<summary>ペルソナが途中で切れている</summary>

```bash
# 品質チェックを実行
python check_persona_completeness.py

# 推奨アクション:
# 1. max_output_tokens の設定を確認
# 2. 入力データを分割して処理
# 3. 再生成を実行
python generate_persona.py --force
```
</details>

## 📚 ドキュメント

### 主要ドキュメント
- [思考プロセス分析（リフレクションノート）](THOUGHT_ANALYSIS_README.md)
- [思考プロセス分析（ドキュメント生成）](DOCUMENT_THOUGHT_ANALYSIS_README.md)
- [RAG技術手法詳細](RAG_TECHNICH.md)
- [CRAG実装レポート](CRAG_IMPLEMENTATION.md)
- [インストールガイド](README_INSTALL.md)

### 対応ファイル形式
- **Google Workspace**: Docs, Slides, Sheets（自動変換）
- **Microsoft Office**: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- **PDF**: テキスト/画像混在対応（OCR機能付き）
- **テキスト**: Plain Text, Markdown
- **画像**: OCR（Gemini Vision API）

## ⚠️ 制限事項

### S3 Vectorsプレビュー版
- 対応リージョン限定（us-west-2, us-east-1等）
- 最大50Mベクトル/インデックス
- ベクトル次元数: 最大2048

### Gemini API
- レート制限: 60 RPM (Flash), 15 RPM (Pro)
- 入力トークン: 最大1Mトークン
- 出力トークン: 最大32,768トークン

### ファイル処理
- 単一ファイル最大サイズ: 100MB（設定可能）
- PDFページ数: 最大500ページ（推奨）

## 🎯 今後の計画

### 短期（1-2ヶ月）
- ✅ AWS S3 Vectors統合（完了）
- ✅ RAG 2段階検索（完了）
- ✅ 思考プロセス分析機能（完了）
- ✅ ペルソナ管理システム（完了）
- ⏳ Slackデータソース統合（開発中）
- 📋 Web UI開発（計画中）

### 中期（3-6ヶ月）
- フィードバックループ機能
- 定期実行スケジューラー
- API化（FastAPI）
- マルチテナント対応

### 長期（6ヶ月以降）
- ベクトルDB拡張（Pinecone/Qdrant対応）
- モデルファインチューニング
- エンタープライズ機能（SSO/監査ログ）

## 📖 参考資料

### 外部ドキュメント
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [AWS S3 Vectors Preview](https://aws.amazon.com/s3/features/s3-express-one-zone/)
- [Unstructured.io Documentation](https://unstructured-io.github.io/unstructured/)
- [LangChain Documentation](https://python.langchain.com/)
