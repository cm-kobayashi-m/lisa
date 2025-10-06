# LISA PoC - リフレクションノート自動生成

Google Driveの「案件情報」フォルダから各案件のファイルを取得し、Gemini APIを使用してリフレクションノートを自動生成するPoCスクリプトです。

## 機能

- Google Drive OAuth 2.0認証
- `/案件情報/` 配下の案件フォルダを自動検出
- 環境変数で処理対象案件を指定 (`*`で全案件、カンマ区切りで個別指定)
- 対応ファイル形式:
  - Google Docs, Slides, Sheets
  - PDF
  - MS Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- Gemini APIによるリフレクションノート生成
- 各案件ごとに `outputs/{案件名}/reflection_note.md` を出力

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

`.env` の内容:

```bash
# Gemini API Key (https://aistudio.google.com/app/apikey から取得)
GEMINI_API_KEY=your-gemini-api-key-here

# 処理対象の案件 (カンマ区切り、*で全案件)
PROJECT_NAMES=*

# 案件情報フォルダのID (Google DriveのURLから取得)
# https://drive.google.com/drive/folders/XXXXXXXXXX ← この部分
PROJECTS_FOLDER_ID=your-projects-folder-id
```

### 4. Google DriveフォルダID取得方法

1. Google Driveで「案件情報」フォルダを開く
2. URLから `folders/` 以降の部分をコピー
   ```
   https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i
                                            ↑ この部分
   ```
3. `.env` の `PROJECTS_FOLDER_ID` に貼り付け

## 使用方法

### 基本実行

```bash
python generate_note.py
```

### 実行例

#### 全案件を処理
```bash
# .envで設定
PROJECT_NAMES=*

# 実行
python generate_note.py
```

#### 特定の案件のみ処理
```bash
# .envで設定
PROJECT_NAMES=案件A,案件B,案件C

# 実行
python generate_note.py
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

### `PROJECTS_FOLDER_ID が設定されていません`

`.env` ファイルに `PROJECTS_FOLDER_ID` (案件情報フォルダのID) を設定してください。

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

- Phase 2: 人間の確認・追加入力フロー
- Phase 3: 完全自動化 (Analyze Engine)
- API化 (FastAPI統合)
- マルチテナント対応
