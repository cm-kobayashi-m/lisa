#!/usr/bin/env python3
"""
LISA PoC - マルチモーダルRAG用ベクトルDB構築スクリプト

GeminiのマルチモーダルAPIを使用して、PDFやWordファイルを直接読み込み、
チャンク化とベクトル化を行い、S3 Vectorsに保存します。

使用方法:
    # すべてのプロジェクトを処理
    python3 generate_rag_multimodal.py

    # 特定のプロジェクトのみ処理
    python3 generate_rag_multimodal.py --project "LISAのPoCテスト"

特徴:
    - Pythonでのテキスト抽出不要（Geminiが直接処理）
    - Geminiによるインテリジェントなチャンク分割
    - マルチモーダル処理（PDF、画像、Word、Excel等）
    - シンプルな実装
"""
import os
import sys
import io
import yaml
import argparse
import gc
import traceback
import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Google Drive API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError

# Gemini API (新SDK)
from google import genai
from google.genai import types

# RAGモジュール
from rag.vector_store import S3VectorStore, Document
from rag.embeddings import GeminiEmbeddings

# 定数
SCOPES = ['https://www.googleapis.com/auth/drive']  # PDF変換にはfullスコープが必要
TOKEN_FILE = 'token.yaml'
CREDENTIALS_FILE = 'credentials.json'
TEMP_DIR = 'temp_files'

# MS Office to Google Workspace MIMEタイプマッピング
OFFICE_TO_GOOGLE_MIME = {
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        'application/vnd.google-apps.document',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        'application/vnd.google-apps.spreadsheet',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        'application/vnd.google-apps.presentation',
    'application/msword':
        'application/vnd.google-apps.document',
    'application/vnd.ms-excel':
        'application/vnd.google-apps.spreadsheet',
    'application/vnd.ms-powerpoint':
        'application/vnd.google-apps.presentation',
}

# RAG設定
VECTOR_BUCKET_NAME = os.getenv('VECTOR_BUCKET_NAME', 'lisa-poc-vectors')
VECTOR_INDEX_NAME = os.getenv('VECTOR_INDEX_NAME', 'project-documents')
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')

# 環境変数読み込み
load_dotenv()


def authenticate():
    """OAuth 2.0認証（Google Drive API用）"""
    creds = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as token:
            token_data = yaml.safe_load(token)
            creds = Credentials(
                token=token_data['token'],
                refresh_token=token_data.get('refresh_token'),
                token_uri=token_data.get('token_uri'),
                client_id=token_data.get('client_id'),
                client_secret=token_data.get('client_secret'),
                scopes=token_data.get('scopes')
            )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"[ERROR] {CREDENTIALS_FILE} が見つかりません。")
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        token_data = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        with open(TOKEN_FILE, 'w') as token:
            yaml.safe_dump(token_data, token, default_flow_style=False)

    return creds


def get_drive_service(creds):
    """Google Drive サービス取得"""
    return build('drive', 'v3', credentials=creds)


def list_project_folders(service, projects_folder_id: str) -> List[Dict[str, str]]:
    """案件情報フォルダ配下のフォルダ一覧を取得"""
    query = f"'{projects_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"

    results = service.files().list(
        q=query,
        fields="files(id, name)",
        orderBy="name"
    ).execute()

    return results.get('files', [])


def list_files_in_folder(service, folder_id: str) -> List[Dict[str, str]]:
    """フォルダ配下のファイル一覧を再帰的に取得"""
    all_files = []

    # 処理対象のMIMEタイプ
    target_mime_types = [
        'application/pdf',
        'application/vnd.google-apps.document',
        'application/vnd.google-apps.spreadsheet',
        'application/vnd.google-apps.presentation',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'image/png',
        'image/jpeg',
        'image/jpg'
    ]

    def _list_recursive(current_folder_id: str):
        """再帰的にファイルとフォルダを取得"""
        query = f"'{current_folder_id}' in parents and trashed=false"

        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, size, createdTime, modifiedTime)",
            pageSize=1000
        ).execute()

        items = results.get('files', [])

        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                _list_recursive(item['id'])
            elif item['mimeType'] in target_mime_types:
                all_files.append(item)

    _list_recursive(folder_id)
    return all_files


def convert_office_via_google_drive(service, file_content: bytes, file_name: str, mime_type: str) -> Optional[Tuple[bytes, str]]:
    """
    Google Drive経由でMS OfficeファイルをPDFに変換

    Returns:
        (PDFコンテンツ, 'application/pdf') または None
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(HttpError)
    )
    def _convert():
        uploaded_file_id = None

        try:
            # Google Workspace形式のMIMEタイプを取得
            google_mime_type = OFFICE_TO_GOOGLE_MIME.get(mime_type)
            if not google_mime_type:
                print(f"    [WARN] 未対応のMIMEタイプ: {mime_type}")
                return None

            # ファイルメタデータ（Google形式を指定してインポート）
            file_metadata = {
                'name': f"temp_{file_name}",
                'mimeType': google_mime_type  # Google形式を指定
            }

            # 専用フォルダIDがある場合は指定（環境変数から取得）
            temp_folder_id = os.getenv('GOOGLE_DRIVE_TEMP_FOLDER_ID')
            if temp_folder_id:
                file_metadata['parents'] = [temp_folder_id]

            # ファイルをアップロード（同時にGoogle形式に変換）
            print(f"    Google Driveにアップロード・変換中...")
            media = MediaIoBaseUpload(
                io.BytesIO(file_content),
                mimetype=mime_type,
                resumable=True
            )

            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,mimeType'
            ).execute()

            uploaded_file_id = uploaded_file.get('id')

            # PDFとしてエクスポート
            print(f"    PDFエクスポート中...")
            request = service.files().export_media(
                fileId=uploaded_file_id,
                mimeType='application/pdf'
            )

            # PDFコンテンツを取得
            pdf_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(pdf_stream, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

            pdf_content = pdf_stream.getvalue()
            print(f"    → PDF変換成功 (Google Drive Cloud)")

            return pdf_content, 'application/pdf'

        except HttpError as e:
            if e.resp.status == 429:  # Rate limit
                print(f"    [WARN] APIレート制限: {e}")
                raise  # リトライ
            elif e.resp.status == 403:  # Permission denied
                print(f"    [ERROR] 権限エラー: {e}")
            else:
                print(f"    [ERROR] Google Drive APIエラー: {e}")
            return None

        except Exception as e:
            print(f"    [ERROR] 予期しないエラー: {e}")
            return None

        finally:
            # アップロードしたファイルを削除（クリーンアップ）
            if uploaded_file_id:
                try:
                    service.files().delete(fileId=uploaded_file_id).execute()
                    print(f"    一時ファイルを削除: {uploaded_file_id}")
                except Exception as e:
                    print(f"    [WARN] 一時ファイル削除失敗: {e}")
                    # 削除失敗をログに記録（後で手動削除が必要）
                    with open('cleanup_failed.log', 'a') as log:
                        log.write(f"{datetime.datetime.now()}: {uploaded_file_id}\n")

    return _convert()


class MultimodalChunker:
    """Geminiマルチモーダルによるチャンク化処理"""

    def __init__(self, gemini_client: genai.Client, embeddings: GeminiEmbeddings):
        self.client = gemini_client
        self.embeddings = embeddings
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    def chunk_file_with_gemini(self, service, file_info: Dict, project_name: str) -> List[Dict]:
        """
        Geminiマルチモーダルでファイルを読み込み、チャンク化する

        Returns:
            チャンク情報のリスト（text, metadata）
        """
        file_id = file_info['id']
        file_name = file_info['name']
        mime_type = file_info['mimeType']

        print(f"  処理中: {file_name}")

        # ファイルサイズチェック
        file_size = int(file_info.get('size', 0))
        if file_size > 20 * 1024 * 1024:
            print(f"    [SKIP] ファイルサイズが大きすぎます: {file_size / 1024 / 1024:.1f}MB")
            return []

        try:
            # ファイルをダウンロードまたはエクスポート（MS OfficeはPDF変換）
            download_result = self._download_file(service, file_id, file_name, mime_type)

            if not download_result:
                print(f"    [SKIP] ファイルを読み込めませんでした")
                return []

            file_content, actual_mime_type = download_result

            # Geminiでチャンク化を実行（実際のMIMEタイプを使用）
            chunks = self._request_chunking(file_content, file_name, actual_mime_type, project_name)

            print(f"    {len(chunks)}個のチャンクを生成")
            return chunks

        except Exception as e:
            print(f"    [ERROR] 処理エラー: {e}")
            return []

    def _download_file(self, service, file_id: str, file_name: str, mime_type: str) -> Optional[Tuple[bytes, str]]:
        """ファイルをダウンロード（MS Officeファイルは自動的にPDF変換）"""
        try:
            # Google形式のファイルはPDFとしてエクスポート
            if mime_type in ['application/vnd.google-apps.document',
                           'application/vnd.google-apps.presentation',
                           'application/vnd.google-apps.spreadsheet']:
                request = service.files().export_media(fileId=file_id, mimeType='application/pdf')
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                return file_stream.getvalue(), 'application/pdf'

            # MS Office形式の場合はGoogle Drive経由でPDF変換
            elif mime_type in OFFICE_TO_GOOGLE_MIME:
                print(f"    MS Officeファイルを検出、PDF変換を実行...")
                # まずファイルをダウンロード
                request = service.files().get_media(fileId=file_id)
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

                file_content = file_stream.getvalue()

                # Google Drive経由でPDF変換
                result = convert_office_via_google_drive(service, file_content, file_name, mime_type)
                if result:
                    pdf_content, pdf_mime = result
                    return pdf_content, pdf_mime
                else:
                    print(f"    [WARN] PDF変換失敗、元のファイルを使用")
                    return file_content, mime_type

            else:
                # その他のファイル（PDF、画像など）はそのままダウンロード
                request = service.files().get_media(fileId=file_id)
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                return file_stream.getvalue(), mime_type

        except Exception as e:
            print(f"    [ERROR] ダウンロードエラー: {e}")
            return None

    def _request_chunking(self, file_content: bytes, file_name: str, mime_type: str, project_name: str) -> List[Dict]:
        """Geminiにチャンク化を依頼"""

        # チャンク化用のプロンプト
        chunk_prompt = f"""
あなたはRAG（Retrieval-Augmented Generation）システム用の文書チャンク化の専門家です。
以下のファイルを意味的にまとまりのある複数のチャンクに分割してください。

【ファイル情報】
- プロジェクト: {project_name}
- ファイル名: {file_name}

【チャンク化の要件】
1. 各チャンクは意味的にまとまりのある単位にする（セクション、段落、トピックなど）
2. 各チャンクは500文字程度を目安とするが、意味の切れ目を優先する
3. 重要な情報が失われないように前後の文脈を少し重複させる
4. 表や図の説明は関連する内容と一緒にチャンクに含める

【出力形式】
以下のJSON形式で出力してください：
{{
  "chunks": [
    {{
      "chunk_index": 0,
      "title": "チャンクのタイトルまたは要約",
      "content": "チャンクの内容（実際のテキスト）",
      "topics": ["トピック1", "トピック2"],
      "importance": "high/medium/low"
    }},
    ...
  ]
}}

重要: 必ず有効なJSONフォーマットで返してください。
"""

        try:
            # ファイルサイズによって送信方法を決定
            print(
                f"    ファイルタイプ: {mime_type}, ファイルサイズ: {len(file_content) / 1024 / 1024:.1f}MB"
            )

            # インラインで送信
            print("    Geminiにインラインで送信中...")
            contents = [
                chunk_prompt,
                {
                    "inline_data": {
                        "data": file_content,
                        "mime_type": mime_type
                    }
                }
            ]

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )

            # レスポンスからJSONを抽出
            response_text = response.text

            # JSONブロックを抽出（```json ... ``` の形式を処理）
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text

            # JSONをパース
            result = json.loads(json_str)

            # チャンク情報を整形
            chunks = []
            for chunk_data in result.get('chunks', []):
                chunks.append({
                    'text': chunk_data.get('content', ''),
                    'metadata': {
                        'project_name': project_name,
                        'file_name': file_name,
                        'chunk_index': chunk_data.get('chunk_index', 0),
                        'title': chunk_data.get('title', ''),
                        'topics': chunk_data.get('topics', []),
                        'importance': chunk_data.get('importance', 'medium')
                    }
                })

            return chunks

        except json.JSONDecodeError as e:
            print(f"    [ERROR] JSON解析エラー: {e}")
            # フォールバック：単純な分割
            return self._simple_chunk(file_content, file_name, project_name)
        except Exception as e:
            print(f"    [ERROR] チャンク化エラー: {e}")
            return []

    def _simple_chunk(self, file_content: bytes, file_name: str, project_name: str) -> List[Dict]:
        """フォールバック用の単純なチャンク分割"""
        try:
            # バイナリをテキストに変換を試みる
            text = file_content.decode('utf-8', errors='ignore')

            chunk_size = 500
            chunks = []

            for i in range(0, len(text), chunk_size - 100):  # 100文字のオーバーラップ
                chunk_text = text[i:i + chunk_size]
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'project_name': project_name,
                            'file_name': file_name,
                            'chunk_index': len(chunks),
                            'title': f"Chunk {len(chunks) + 1}",
                            'topics': [],
                            'importance': 'medium'
                        }
                    })

            return chunks
        except:
            return []


class VectorDBBuilder:
    """ベクトルDB構築処理"""

    def __init__(self):
        # Embeddings初期化
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY が設定されていません")

        self.embeddings = GeminiEmbeddings(api_key=api_key)

        # Vector Store初期化
        self.vector_store = S3VectorStore(
            vector_bucket_name=VECTOR_BUCKET_NAME,
            index_name=VECTOR_INDEX_NAME,
            dimension=768,
            distance_metric="cosine",
            region_name=AWS_REGION,
            create_if_not_exists=True
        )

    def save_chunks_to_vector_db(self, chunks: List[Dict], file_name: str) -> int:
        """チャンクをベクトル化してS3 Vectorsに保存"""

        if not chunks:
            return 0

        documents = []

        for chunk_info in chunks:
            text = chunk_info['text']
            metadata = chunk_info['metadata']

            # ドキュメントキー生成
            import hashlib
            key_string = f"{metadata['project_name']}/{file_name}/{metadata['chunk_index']}"
            doc_key = f"doc_{hashlib.md5(key_string.encode()).hexdigest()[:16]}_{metadata['chunk_index']}"

            # Document作成
            doc = Document(
                key=doc_key,
                text=text,
                metadata=metadata
            )

            # ベクトル化
            try:
                doc.vector = self.embeddings.embed_text(text)
                if doc.vector and len(doc.vector) == self.embeddings.dimension:
                    documents.append(doc)
            except Exception as e:
                print(f"    [WARN] ベクトル化エラー: {e}")

        # ベクトルストアに保存
        if documents:
            added = self.vector_store.add_documents(documents, batch_size=5)
            return added

        return 0


def main():
    """メイン処理"""
    print("=" * 60)
    print("LISA PoC - マルチモーダルRAG ベクトルDB構築")
    print("=" * 60)
    print()
    print("特徴:")
    print("  - GeminiマルチモーダルAPIでファイルを直接処理")
    print("  - インテリジェントなチャンク分割")
    print("  - シンプルな実装")
    print()

    # コマンドライン引数
    parser = argparse.ArgumentParser(description='マルチモーダルRAG ベクトルDB構築')
    parser.add_argument('--project', type=str, help='特定のプロジェクトのみ処理')
    args = parser.parse_args()

    # 環境変数チェック
    api_key = os.getenv('GEMINI_API_KEY')
    projects_folder_id = os.getenv('PROJECTS_FOLDER_ID')
    project_names = os.getenv('PROJECT_NAMES', '*')

    if not api_key:
        print("[ERROR] GEMINI_API_KEY が設定されていません")
        sys.exit(1)

    if not projects_folder_id:
        print("[ERROR] PROJECTS_FOLDER_ID が設定されていません")
        sys.exit(1)

    # 認証
    print("[INFO] OAuth認証中...")
    creds = authenticate()
    service = get_drive_service(creds)
    print("[INFO] 認証完了")

    # Geminiクライアント初期化
    print("[INFO] Gemini APIクライアント初期化中...")
    gemini_client = genai.Client(api_key=api_key)
    print("[INFO] 初期化完了")

    # 処理コンポーネント初期化
    vector_db = VectorDBBuilder()
    chunker = MultimodalChunker(gemini_client, vector_db.embeddings)

    # プロジェクト一覧取得
    all_projects = list_project_folders(service, projects_folder_id)

    # フィルタリング
    if args.project:
        target_projects = [p for p in all_projects if p['name'] == args.project]
    else:
        if project_names != '*':
            target_names = [name.strip() for name in project_names.split(',')]
            target_projects = [p for p in all_projects if p['name'] in target_names]
        else:
            target_projects = all_projects

    if not target_projects:
        print("[WARN] 処理対象のプロジェクトが見つかりませんでした")
        sys.exit(0)

    print(f"[INFO] 処理対象: {', '.join([p['name'] for p in target_projects])} ({len(target_projects)}件)")
    print()

    # 各プロジェクトを処理
    total_docs = 0

    for project in target_projects:
        project_name = project['name']
        project_id = project['id']

        print(f"\n{'=' * 50}")
        print(f"プロジェクト: {project_name}")
        print(f"{'=' * 50}")

        # ファイル一覧取得
        files = list_files_in_folder(service, project_id)
        print(f"[INFO] {len(files)}個のファイルを発見")

        project_docs = 0

        for i, file_info in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]", end=" ")

            # Geminiでチャンク化
            chunks = chunker.chunk_file_with_gemini(service, file_info, project_name)

            if chunks:
                # ベクトルDBに保存
                added = vector_db.save_chunks_to_vector_db(chunks, file_info['name'])
                project_docs += added
                print(f"    {added}個のドキュメントを保存")

            # メモリ解放
            gc.collect()

        total_docs += project_docs
        print(f"\n[INFO] {project_name}: {project_docs}個のドキュメントを処理")

    print()
    print("=" * 60)
    print("処理完了")
    print("=" * 60)
    print(f"総ドキュメント数: {total_docs}個")
    print()


if __name__ == "__main__":
    main()
