# LISA PoC - AI駆動型プロジェクト分析システム

プロジェクトのドキュメントをRAG (Retrieval-Augmented Generation) とGoogle Gemini APIを使用して分析し、インサイトやリフレクションノートを自動生成するシステムです。

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

## 主要機能

### 🔍 高度なRAG（Retrieval-Augmented Generation）機能
- **AWS S3 Vectors** を使用した高速ベクトル検索（サーバーレス）
- **2段階検索アルゴリズム**：
  1. プロジェクト名からキーワード生成
  2. キーワードベースで類似プロジェクトを検索
- **ハイブリッドスコアリング**: 類似度と時系列の重み付け統合
- **時系列重み付け**: 最新情報を優先的に検索
- **バッチ処理最適化**: 埋め込み生成とS3保存の並列処理
- **インテリジェントキャッシュ**: 処理済みファイルの自動スキップ

### 📁 マルチデータソース対応
- **Google Drive**: 複数フォルダからの一括取得（実装済み）
- **Slack**: チャンネルメッセージの取得（開発予定）
- **Backlog**: 課題・Wiki情報の取得（開発予定）
- **YAML設定**: `project_config.yaml`による柔軟なプロジェクト管理

### 📄 対応ファイル形式
- **Google Workspace**: Docs, Slides, Sheets (自動変換)
- **Microsoft Office**: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- **PDF**: テキスト/画像混在対応（OCR機能付き）
- **テキスト形式**: Plain Text, Markdown
- **画像からのOCR**: Gemini Vision APIによる文字認識

## セットアップ

### 1. 依存ライブラリのインストール

```bash
cd lisa/poc
pip install -r requirements.txt
```

主要な依存ライブラリ:
- `google-genai`: Google Gemini API SDK（新版）
- `unstructured[pdf,md,docx,xlsx,pptx]`: ファイル解析
- `langchain-google-genai`: LangChain Gemini統合
- `boto3`: AWS S3 Vectors連携
- `func-timeout`: タイムアウト処理
- その他: PyPDF2, python-docx, openpyxl, python-pptx

### 2. Google Cloud Console設定

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. プロジェクト作成 (既存の場合はスキップ)
3. 「APIとサービス」→「ライブラリ」→「Google Drive API」を有効化
4. 「APIとサービス」→「認証情報」
5. 「認証情報を作成」→「OAuthクライアントID」
6. アプリケーションの種類: 「デスクトップアプリ」
7. `credentials.json` をダウンロード → `lisa/poc/credentials.json` に配置

### 3. 環境変数設定

```bash
# .env.exampleをコピー
cp .env.example .env

# .envを編集
vi .env
```

#### 必須の環境変数

```bash
# Google Gemini API Key（必須）
GEMINI_API_KEY=your-gemini-api-key-here

# Geminiモデル選択（デフォルト: gemini-2.5-flash）
GEMINI_MODEL=gemini-2.5-flash  # または gemini-2.5-pro
```

#### AWS S3 Vectors設定（RAG機能使用時に必須）

```bash
# AWSリージョン（S3 Vectors対応: us-west-2, us-east-1等）
AWS_REGION=us-west-2

# S3ベクトルバケット名とインデックス名
VECTOR_BUCKET_NAME=lisa-poc-vectors
VECTOR_INDEX_NAME=project-documents

# 埋め込みモデル設定
EMBEDDING_MODEL=gemini-embedding-001
DIMENSION=1536

# AWS認証（オプション、特定プロファイル使用時）
# AWS_PROFILE=your-profile-name
```

#### RAG検索チューニング（オプション）

```bash
# 基本検索設定
RAG_MIN_SCORE=0.6                  # 最小類似度スコア (0.0-1.0)
RAG_SCORING_METHOD=hybrid           # hybrid/reranking/time_decay/none
RAG_TIME_WEIGHT=0.2                 # 時系列重み (0.0-1.0)
RAG_DECAY_DAYS=90                   # 時間減衰の半減期（日数）

# RAG専用モード時の設定（より多くの文書を検索）
RAG_ONLY_MODE_K_CURRENT=15         # 現在のプロジェクトから取得
RAG_ONLY_MODE_K_SIMILAR=15         # 類似プロジェクトから取得
RAG_ONLY_MODE_MAX_TOTAL=30         # 合計最大件数
RAG_ONLY_MODE_MIN_SCORE=0.3        # RAG専用モード時の最小スコア
```

### 4. プロジェクト設定ファイルの作成

```bash
# サンプルファイルをコピー
cp project_config.yaml.sample project_config.yaml

# 設定ファイルを編集
vi project_config.yaml
```

`project_config.yaml` の例:

```yaml
projects:
  プロジェクトA:
    google_drive:
      - "フォルダID1"  # プロジェクトAのメインフォルダ
      - "フォルダID2"  # プロジェクトAの追加資料フォルダ

  プロジェクトB:
    google_drive:
      - "フォルダID3"
```

Google DriveフォルダIDの取得方法:
1. Google Driveでフォルダを開く
2. URLから `folders/` 以降の部分をコピー
   ```
   https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i
                                            ↑ この部分
   ```

## 使用方法

### 1. RAGインデックスの構築（初回および更新時）

```bash
# 全プロジェクトのインデックスを構築
python3 generate_rag_unstructured.py

# 特定のプロジェクトのみ処理
python3 generate_rag_unstructured.py --project "プロジェクトA"

# キャッシュをクリアして再構築
python3 generate_rag_unstructured.py --clear-cache

# 詳細ログを表示
python3 generate_rag_unstructured.py --verbose
```

### 2. リフレクションノートの生成

```bash
# 基本的な使用方法
python3 generate_note.py

# プロジェクトとタスクを指定
python3 generate_note.py --project "プロジェクトA" --task "機能追加の検討"

# RAG専用モードで実行（より多くの文書を検索）
python3 generate_note.py --project "プロジェクトA" --task "機能追加の検討" --rag-only

# 検索件数を指定
python3 generate_note.py --project "プロジェクトA" --k-current 10 --k-similar 10
```

### 3. S3 Vectorsの管理

```bash
# インデックスのみ削除
python3 delete_s3_vectors.py --index-only

# バケットとインデックスを削除
python3 delete_s3_vectors.py

# 特定のバケット/インデックスを削除
python3 delete_s3_vectors.py --bucket my-bucket --index my-index

# Dry-runモード（削除せず確認のみ）
python3 delete_s3_vectors.py --dry-run
```

### 初回実行時

初回実行時はブラウザが開き、Google認証が求められます:

1. Googleアカウントでログイン
2. 「このアプリは確認されていません」と表示される場合:
   - 「詳細」→「(アプリ名)に移動」をクリック
3. 権限を承認
4. 認証情報が `token.pickle` に保存され、次回以降は自動ログイン

## 出力

```
lisa/poc/outputs/
├── 案件A/
│   └── reflection_note.md
├── 案件B/
│   └── reflection_note.md
└── 案件C/
    └── reflection_note.md
```

## 実行ログ例

```
[INFO] LISA PoC - リフレクションノート自動生成

[INFO] OAuth認証中...
[INFO] OAuth認証完了
[INFO] 案件情報フォルダ: /案件情報/ (ID: 1a2b3c4d5e)
[INFO] 処理対象案件: 案件A, 案件B, 案件C (3件)

[INFO] === 案件A の処理開始 ===
[INFO] ファイル取得: 5件
  - 要件定義書.docx (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
  - 議事録.pdf (application/pdf)
  - 設計書.gdoc (application/vnd.google-apps.document)
  - データ.xlsx (application/vnd.openxmlformats-officedocument.spreadsheetml.sheet)
  - プレゼン.pptx (application/vnd.openxmlformats-officedocument.presentationml.presentation)
[INFO] テキスト抽出完了: 3,245文字
[INFO] Gemini APIでノート生成中...
[INFO] 出力: outputs/案件A/reflection_note.md
[INFO] === 案件A の処理完了 ===

[INFO] === 案件B の処理開始 ===
...

========================================
処理サマリー
========================================
成功: 3件
失敗: 0件
========================================
```

## トラブルシューティング

### 📝 基本的なエラー対処

#### `credentials.json が見つかりません`
Google Cloud Consoleから `credentials.json` をダウンロードし、`lisa/poc/` に配置してください。

#### `GEMINI_API_KEY が設定されていません`
`.env` ファイルに `GEMINI_API_KEY` を設定してください。APIキーは [Google AI Studio](https://aistudio.google.com/app/apikey) から取得できます。

#### `project_config.yaml が見つかりません`
```bash
# サンプルファイルから作成
cp project_config.yaml.sample project_config.yaml
# プロジェクト設定を編集
vi project_config.yaml
```

### 🔧 AWS S3 Vectors関連

#### S3 Vectorsのアクセスエラー
```bash
# AWS CLIの設定を確認
aws configure list

# 特定のプロファイルを使用する場合
export AWS_PROFILE=your-profile-name

# リージョンが正しいか確認（S3 Vectors対応リージョン）
export AWS_REGION=us-west-2  # または us-east-1, us-east-2等
```

#### ベクトル検索が機能しない
```bash
# インデックスの再構築
python3 generate_rag_unstructured.py --clear-cache

# S3 Vectorsのステータス確認
aws s3api head-bucket --bucket $VECTOR_BUCKET_NAME
```

### ⏱️ パフォーマンス関連

#### ファイルの処理が終わらない
大きなファイル（Excel、PDF、Word等）で処理が長時間かかる場合：

```bash
# .envファイルでタイムアウト設定を調整（デフォルト: 300秒）
PARTITION_TIMEOUT=600

# PDFの処理戦略を変更（hi_res無効化）
PDF_USE_HI_RES=false
```

#### メモリ不足エラー
```bash
# バッチサイズを調整
BATCH_SIZE=10  # デフォルト: 20

# 並列処理数を制限
MAX_WORKERS=2  # デフォルト: 4
```

### 🔐 認証関連

#### Google認証エラー (`token.yaml`)
```bash
# トークンを削除して再認証
rm token.yaml
python3 generate_note.py
```

#### Gemini APIのレート制限
```bash
# .envでモデルを変更（より低コストなモデルへ）
GEMINI_MODEL=gemini-2.5-flash-lite

# リトライ設定を調整
MAX_RETRIES=5
RETRY_DELAY=2
```

## 📊 RAG検索フローの詳細

### 2段階検索アルゴリズム

```
1. キーワード生成フェーズ
   プロジェクト名 → Gemini API → 関連キーワード抽出
   例: "EC売上改善" → "EC 電子商取引 売上 分析 最適化..."

2. ベクトル検索フェーズ
   a) 現在のプロジェクトから検索（k=10〜15件）
   b) キーワードで類似プロジェクトを検索
   c) 類似プロジェクトから追加検索（k=10〜15件）

3. スコアリング＆ランキング
   - コサイン類似度: 0.8 (80%)
   - 時系列スコア: 0.2 (20%)
   → ハイブリッドスコア = 0.8 × 類似度 + 0.2 × 時間スコア
```

### ハイブリッドスコアリング方式

| 方式 | 説明 | 使用場面 |
|------|------|----------|
| **hybrid** (推奨) | 類似度と時間を重み付け統合 | バランス重視 |
| **reranking** | 類似度検索後、時間でソート | 新しさ優先 |
| **time_decay** | 類似度に時間減衰を適用 | 古い情報を段階的に除外 |
| **none** | 類似度のみ | 時系列を考慮しない |

## 制限事項

### 🚧 現在の制限

- **S3 Vectorsプレビュー版の制限**:
  - 対応リージョン限定（us-west-2, us-east-1等）
  - 最大50Mベクトル/インデックス
  - ベクトル次元数: 最大2048

- **Gemini API制限**:
  - レート制限: 60 RPM (Flash), 15 RPM (Pro)
  - 入力トークン: 最大1Mトークン
  - 埋め込み生成: バッチあたり最大100テキスト

- **ファイル処理制限**:
  - 単一ファイル最大サイズ: 100MB（設定可能）
  - PDFページ数: 最大500ページ（推奨）

### ⚠️ 既知の問題

- 大量の画像を含むPDFでメモリ使用量が増大
- ネットワーク不安定時のリトライ処理が不完全
- 日本語以外の言語でのOCR精度低下

## 次のステップ

### 🎯 短期計画（1-2ヶ月）
- ✅ AWS S3 Vectors統合（完了）
- ✅ RAG 2段階検索（完了）
- ⏳ Slackデータソース統合（開発中）
- ⏳ Backlogデータソース統合（計画中）
- 📋 Web UI開発（Streamlit/Gradio）

### 🚀 中期計画（3-6ヶ月）
- **Phase 2**: 人間のフィードバックループ
  - レビュー＆承認フロー
  - フィードバック学習機能
- **Phase 3**: 完全自動化（Analyze Engine）
  - 定期実行スケジューラー
  - 差分検出＆増分更新
- **API化**: FastAPI/GraphQL統合
- **マルチテナント**: 組織別の分離管理

### 🌟 長期計画（6ヶ月以降）
- **ベクトルDB拡張**:
  - Pinecone/Qdrant/Weaviate対応
  - ハイブリッド検索（キーワード＋ベクトル）
- **モデル最適化**:
  - ファインチューニング
  - プロンプト最適化AI
- **エンタープライズ機能**:
  - SSO/SAML認証
  - 監査ログ＆コンプライアンス
  - ロールベースアクセス制御

## 📚 参考資料

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [AWS S3 Vectors Preview](https://aws.amazon.com/s3/features/s3-express-one-zone/)
- [Unstructured.io Documentation](https://unstructured-io.github.io/unstructured/)
- [LangChain Documentation](https://python.langchain.com/)
