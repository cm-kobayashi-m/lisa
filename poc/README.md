# LISA PoC - AI駆動型プロジェクト分析システム

プロジェクトのドキュメントをRAG (Retrieval-Augmented Generation) とGemini APIを使用して分析し、インサイトやリフレクションノートを自動生成するシステムです。

## 主要機能

### RAG（Retrieval-Augmented Generation）機能
- AWS S3 Vectors を使用した高速ベクトル検索
- プロジェクト横断での類似情報検索
- 時系列重み付けによる最新情報の優先
- キャッシュ機能による高速化

### マルチデータソース対応
- 複数のGoogle Driveフォルダからデータ取得
- 将来的にSlack、Backlogのデータソースにも対応予定
- YAMLベースの柔軟なプロジェクト設定

### 対応ファイル形式
- Google Docs, Slides, Sheets
- PDF
- MS Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- テキストファイル、Markdownファイル

## セットアップ

### 1. 依存ライブラリのインストール

```bash
cd lisa/poc
pip install -r requirements.txt
```

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

主要な環境変数:

```bash
# Gemini API Key (必須)
GEMINI_API_KEY=your-gemini-api-key-here

# AWS設定 (S3 Vectors使用時に必須)
AWS_REGION=us-west-2
VECTOR_BUCKET_NAME=lisa-poc-vectors
VECTOR_INDEX_NAME=project-documents

# RAG検索設定 (オプション)
RAG_MIN_SCORE=0.6
RAG_SCORING_METHOD=hybrid
RAG_TIME_WEIGHT=0.2
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
python generate_rag_unstructured.py

# 特定のプロジェクトのみ
python generate_rag_unstructured.py --project "プロジェクトA"
```

### 2. リフレクションノートの生成

```bash
# プロジェクトとタスクを指定して実行
python generate_note.py --project "プロジェクトA" --task "機能追加の検討"

# RAG専用モードで実行（より多くの文書を検索）
python generate_note.py --project "プロジェクトA" --task "機能追加の検討" --rag-only
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

### `credentials.json が見つかりません`

Google Cloud Consoleから `credentials.json` をダウンロードし、`lisa/poc/` に配置してください。

### `GEMINI_API_KEY が設定されていません`

`.env` ファイルに `GEMINI_API_KEY` を設定してください。APIキーは [Google AI Studio](https://aistudio.google.com/app/apikey) から取得できます。

### `project_config.yaml が見つかりません`

`project_config.yaml.sample` をコピーして `project_config.yaml` を作成し、プロジェクト設定を記述してください。

### ファイルの処理が終わらない

大きなファイル（Excel、PDF、Word等）で処理が長時間かかる場合は、タイムアウト設定を調整してください。詳細は [FILE_TIMEOUT_GUIDE.md](FILE_TIMEOUT_GUIDE.md) を参照してください。

```bash
# .envファイルで設定（デフォルト: 300秒）
PARTITION_TIMEOUT=600
```

### 認証エラー (`token.pickle`)

`token.pickle` を削除して再実行すると、再度ブラウザ認証が行われます:

```bash
rm token.pickle
python generate_note.py
```

## 制限事項

- PoCのため、エラーハンドリングは最小限です
- 大量のファイル処理時はAPI制限に注意してください
- Gemini APIの入力トークン制限により、大量のテキストは切り詰められる可能性があります

## 次のステップ

このPoCで実現可能性を検証後、以下の機能拡張を検討:

### 短期計画
- Slackデータソースの統合
- Backlogデータソースの統合
- Web UIの開発

### 中期計画
- Phase 2: 人間の確認・追加入力フロー
- Phase 3: 完全自動化 (Analyze Engine)
- API化 (FastAPI統合)
- マルチテナント対応

### 長期計画
- 他のベクトルDBへの対応（Pinecone、Qdrant等）
- ファインチューニング済みモデルの活用
- エンタープライズ向け機能（監査ログ、権限管理等）
