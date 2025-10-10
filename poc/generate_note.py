#!/usr/bin/env python3
"""
LISA PoC - リフレクションノート自動生成スクリプト（Google Drive Cloud PDF版）

外部バイナリ不要：Google Drive APIを使用してクラウド上でMS OfficeファイルをPDF変換し、
Gemini Files APIでレイアウトを保持したまま分析します。

使用方法:
    python generate_note_cloud_pdf.py
"""
import datetime
import os
import sys
import io
import yaml
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Google Drive API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError

# ファイル抽出（フォールバック用）
import PyPDF2
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

# Gemini API (新SDK)
from google import genai
from google.genai import types

# 定数
SCOPES = ['https://www.googleapis.com/auth/drive']  # PDF変換にはfullスコープが必要
TOKEN_FILE = 'token.yaml'
CREDENTIALS_FILE = 'credentials.json'
OUTPUT_DIR = 'outputs'
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

# 環境変数読み込み
load_dotenv()


def authenticate():
    """OAuth 2.0認証（Google Drive API用）- 最小権限スコープ"""
    creds = None

    # token.yamlが存在する場合は読み込み
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

    # 認証情報が無効または存在しない場合は再認証
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"[ERROR] {CREDENTIALS_FILE} が見つかりません。")
                print("Google Cloud Consoleからダウンロードして配置してください。")
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # token.yamlに保存
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

    folders = results.get('files', [])
    return folders


def filter_projects(all_projects: List[Dict[str, str]], project_names: str) -> List[Dict[str, str]]:
    """環境変数で指定された案件でフィルタリング"""
    if project_names == "*":
        return all_projects

    target_names = [name.strip() for name in project_names.split(',')]
    filtered = [p for p in all_projects if p['name'] in target_names]

    return filtered


def list_files_in_folder(service, folder_id: str) -> List[Dict[str, str]]:
    """フォルダ配下のファイル一覧を再帰的に取得 (ネストしたフォルダも含む)"""
    all_files = []

    def _list_recursive(current_folder_id: str):
        """再帰的にファイルとフォルダを取得"""
        query = f"'{current_folder_id}' in parents and trashed=false"

        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType)",
            pageSize=1000
        ).execute()

        items = results.get('files', [])

        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                # フォルダの場合は再帰的に探索
                _list_recursive(item['id'])
            else:
                # ファイルの場合はリストに追加
                all_files.append(item)

    _list_recursive(folder_id)
    return all_files


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(HttpError)
)
def convert_office_via_google_drive(service, file_content: bytes, file_name: str, mime_type: str) -> Optional[Tuple[Path, str]]:
    """
    Google Drive経由でMS OfficeファイルをPDFに変換

    Args:
        service: Google Drive APIサービス
        file_content: ファイルコンテンツ（バイナリ）
        file_name: ファイル名
        mime_type: 元ファイルのMIMEタイプ

    Returns:
        (PDFファイルパス, 'application/pdf') または None
    """
    temp_dir = Path(TEMP_DIR)
    temp_dir.mkdir(exist_ok=True)

    uploaded_file_id = None

    try:
        # Google Workspace形式のMIMEタイプを取得
        google_mime_type = OFFICE_TO_GOOGLE_MIME.get(mime_type)
        if not google_mime_type:
            print(f"[WARN] 未対応のMIMEタイプ: {mime_type}")
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
        print(f"    アップロード完了: {uploaded_file_id}")

        # PDFとしてエクスポート
        print(f"    PDFエクスポート中...")
        request = service.files().export_media(
            fileId=uploaded_file_id,
            mimeType='application/pdf'
        )

        # PDFファイルを一時ディレクトリに保存
        pdf_path = temp_dir / f"{Path(file_name).stem}.pdf"
        with open(pdf_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"    ダウンロード進捗: {int(status.progress() * 100)}%")

        print(f"    → PDF変換成功 (Google Drive Cloud)")
        return pdf_path, 'application/pdf'

    except HttpError as e:
        if e.resp.status == 429:  # Rate limit
            print(f"[WARN] APIレート制限: {e}")
            raise  # リトライ
        elif e.resp.status == 403:  # Permission denied
            print(f"[ERROR] 権限エラー: {e}")
        else:
            print(f"[ERROR] Google Drive APIエラー: {e}")
        return None

    except Exception as e:
        print(f"[ERROR] 予期しないエラー: {e}")
        return None

    finally:
        # アップロードしたファイルを削除（クリーンアップ）
        if uploaded_file_id:
            try:
                service.files().delete(fileId=uploaded_file_id).execute()
                print(f"    一時ファイルを削除: {uploaded_file_id}")
            except Exception as e:
                print(f"[WARN] 一時ファイル削除失敗: {e}")
                # 削除失敗をログに記録（後で手動削除が必要）
                with open('cleanup_failed.log', 'a') as log:
                    log.write(f"{datetime.datetime.now()}: {uploaded_file_id}\n")


def extract_text_from_file(service, file: Dict[str, str]) -> str:
    """フォールバック：ファイル形式に応じてテキスト抽出"""
    file_id = file['id']
    file_name = file['name']
    mime_type = file['mimeType']

    print(f"  - {file_name} ({mime_type}) - テキスト抽出モード")

    # Google形式
    if mime_type == 'application/vnd.google-apps.document':
        try:
            request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            file_content = request.execute()
            return file_content.decode('utf-8')
        except Exception as e:
            print(f"[WARN] Google Docs抽出エラー: {e}")
            return ""

    elif mime_type == 'application/vnd.google-apps.spreadsheet':
        try:
            request = service.files().export_media(fileId=file_id, mimeType='text/csv')
            file_content = request.execute()
            return file_content.decode('utf-8')
        except Exception as e:
            print(f"[WARN] Google Sheets抽出エラー: {e}")
            return ""

    elif mime_type == 'application/vnd.google-apps.presentation':
        try:
            request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            file_content = request.execute()
            return file_content.decode('utf-8')
        except Exception as e:
            print(f"[WARN] Google Slides抽出エラー: {e}")
            return ""

    # バイナリファイルをダウンロード
    try:
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        content = file_stream.getvalue()
    except Exception as e:
        print(f"[WARN] ダウンロードエラー ({file_name}): {e}")
        return ""

    # テキスト抽出
    if mime_type == 'application/pdf':
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"[WARN] PDF抽出エラー: {e}")
            return ""

    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        try:
            doc = Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"[WARN] Word抽出エラー: {e}")
            return ""

    elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        try:
            wb = load_workbook(io.BytesIO(content), data_only=True)
            text = ""
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    text += "\t".join([str(cell) if cell is not None else "" for cell in row]) + "\n"
            return text
        except Exception as e:
            print(f"[WARN] Excel抽出エラー: {e}")
            return ""

    elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        try:
            prs = Presentation(io.BytesIO(content))
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        except Exception as e:
            print(f"[WARN] PowerPoint抽出エラー: {e}")
            return ""

    else:
        print(f"[WARN] 未対応の形式: {mime_type}")
        return ""


def download_and_convert_file(service, file: Dict[str, str]) -> Optional[Tuple[Path, str]]:
    """ファイルをダウンロードし、必要に応じてPDFに変換"""
    file_id = file['id']
    file_name = file['name']
    mime_type = file['mimeType']

    # 一時ディレクトリを作成
    temp_dir = Path(TEMP_DIR)
    temp_dir.mkdir(exist_ok=True)

    print(f"  - {file_name} ({mime_type})")

    # 変換モードを環境変数から取得
    conversion_mode = os.getenv('CONVERSION_MODE', 'cloud_pdf')

    # Google形式のファイルは直接PDFエクスポート
    if mime_type in [
        'application/vnd.google-apps.document',
        'application/vnd.google-apps.presentation',
        'application/vnd.google-apps.spreadsheet'
    ]:
        try:
            request = service.files().export_media(fileId=file_id, mimeType='application/pdf')
            output_path = temp_dir / f"{file_name}.pdf"
            with open(output_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            print(f"    → PDFエクスポート成功 (Google形式)")
            return output_path, 'application/pdf'
        except Exception as e:
            print(f"[WARN] Google PDFエクスポートエラー: {e}")
            return None, None

    # MS Office形式の処理
    if mime_type in OFFICE_TO_GOOGLE_MIME and conversion_mode == 'cloud_pdf':
        # Google Drive経由でPDF変換
        try:
            # ファイルをダウンロード
            request = service.files().get_media(fileId=file_id)
            file_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(file_stream, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            content = file_stream.getvalue()

            # Google Drive経由でPDF変換
            result = convert_office_via_google_drive(service, content, file_name, mime_type)
            if result:
                return result
            else:
                print(f"[WARN] Cloud PDF変換失敗、テキスト抽出にフォールバック")
                # フォールバックは後述
        except Exception as e:
            print(f"[ERROR] ファイル処理エラー: {e}")
            return None, None

    # その他のファイル（PDFや画像）はそのまま保存
    if mime_type in ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg']:
        try:
            request = service.files().get_media(fileId=file_id)
            file_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(file_stream, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            content = file_stream.getvalue()

            output_path = temp_dir / file_name
            with open(output_path, 'wb') as f:
                f.write(content)
            return output_path, mime_type
        except Exception as e:
            print(f"[WARN] ダウンロードエラー: {e}")
            return None, None

    print(f"[WARN] 未対応の形式: {mime_type}")
    return None, None


class GeminiQuotaError(Exception):
    """Gemini APIのクォータ制限エラー"""
    pass


def _is_quota_error(exception: Exception) -> bool:
    """クォータエラーかどうかを判定"""
    error_msg = str(exception)
    return '429' in error_msg or 'quota' in error_msg.lower()


def initialize_gemini_client() -> genai.Client:
    """Gemini APIクライアントを初期化"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("[ERROR] GEMINI_API_KEY が環境変数に設定されていません。")
        sys.exit(1)

    return genai.Client(api_key=api_key)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(GeminiQuotaError),
    before_sleep=lambda retry_state: print(
        f"[WARN] クォータ制限検出 (試行 {retry_state.attempt_number}/5)"
        f" - {retry_state.next_action.sleep}秒待機してリトライします..."
    )
)
def analyze_file_with_gemini(client: genai.Client, file_path: Optional[Path], file_name: str,
                            mime_type: Optional[str], text_content: Optional[str] = None,
                            use_rag: bool = True, project_name: Optional[str] = None) -> str:
    """Gemini Files APIまたはテキストモードでファイルを分析（RAG拡張版）"""
    # 改善版のプロンプトを使用
    from improved_prompts import get_analyze_file_prompt
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    # RAGコンテキストの取得
    rag_context = ""
    if use_rag and os.getenv('USE_RAG', 'true').lower() == 'true':
        try:
            print(f"[INFO] RAG検索を実行中...")
            # RAGRetrieverの初期化
            from rag.rag_retriever import RAGRetriever
            from rag.vector_store import S3VectorStore
            from rag.embeddings import GeminiEmbeddings

            # 初期化
            embeddings = GeminiEmbeddings(api_key=os.getenv('GEMINI_API_KEY'))
            vector_store = S3VectorStore(
                vector_bucket_name=os.getenv('VECTOR_BUCKET_NAME', 'lisa-poc-vectors'),
                index_name=os.getenv('VECTOR_INDEX_NAME', 'project-documents'),
                dimension=768,
                region_name=os.getenv('AWS_REGION', 'us-west-2'),
                create_if_not_exists=False  # 既存のインデックスを使用
            )
            retriever = RAGRetriever(vector_store, embeddings)

            # ファイル名とプロジェクト名から検索クエリを生成
            search_query = f"{file_name}"
            if project_name:
                search_query = f"{project_name} {file_name}"

            # テキストコンテンツがある場合は、その一部も検索クエリに含める
            if text_content and len(text_content) > 100:
                search_query += f" {text_content[:500]}"

            # 類似ドキュメント検索
            results = retriever.search_similar_documents(
                query=search_query,
                project_name=project_name,
                k=5
            )

            if results:
                rag_context = retriever.format_context_for_prompt(results, max_chars=3000)
                print(f"[INFO] RAGから{len(results)}件の関連情報を取得")
            else:
                print(f"[INFO] RAG検索結果なし")

        except Exception as e:
            print(f"[WARN] RAG検索でエラーが発生しました: {e}")
            # RAGが失敗しても処理は継続

    try:
        # プロンプト
        base_prompt = get_analyze_file_prompt(file_name)

        # RAGコンテキストをプロンプトに追加
        if rag_context:
            base_prompt = f"{base_prompt}\n\n{rag_context}"

        # ファイルパスがある場合（PDF/画像）
        if file_path and file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"    ファイルサイズ: {file_size_mb:.2f} MB")

            # インラインで送信
            print(f"    インラインでファイルを送信中...")
            with open(file_path, 'rb') as f:
                file_data = f.read()

            contents = [
                base_prompt,
                {
                    "inline_data": {
                        "data": file_data,
                        "mime_type": mime_type
                    }
                }
            ]

            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )

        # テキストコンテンツの場合
        elif text_content:
            print(f"    テキストモードで分析中...")
            # テキストコンテンツを基本プロンプトと結合
            full_prompt = f"""{base_prompt}

# ファイル内容（テキスト抽出済み）:
{text_content[:10000]}  # 最大10000文字に制限
"""
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt
            )

        else:
            return "分析対象のコンテンツがありません"

        return response.text

    except Exception as e:
        if _is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[ERROR] Gemini API呼び出しエラー: {e}")
            raise


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(GeminiQuotaError),
    before_sleep=lambda retry_state: print(
        f"[WARN] クォータ制限検出 (試行 {retry_state.attempt_number}/5)"
        f" - {retry_state.next_action.sleep}秒待機してリトライします..."
    )
)
def generate_final_reflection_note(client: genai.Client, project_name: str, file_summaries: List[Dict[str, str]],
                                  use_rag: bool = True) -> tuple[str, str]:
    """全ファイル分析結果から最終的なリフレクションノートを生成（RAG拡張版）"""

    # RAGから過去の類似プロジェクト情報を取得
    rag_context = ""
    if use_rag and os.getenv('USE_RAG', 'true').lower() == 'true':
        try:
            print(f"[INFO] 類似プロジェクト情報をRAGから検索中...")
            from rag.rag_retriever import RAGRetriever
            from rag.vector_store import S3VectorStore
            from rag.embeddings import GeminiEmbeddings

            # 初期化
            embeddings = GeminiEmbeddings(api_key=os.getenv('GEMINI_API_KEY'))
            vector_store = S3VectorStore(
                vector_bucket_name=os.getenv('VECTOR_BUCKET_NAME', 'lisa-poc-vectors'),
                index_name=os.getenv('VECTOR_INDEX_NAME', 'project-documents'),
                dimension=768,
                region_name=os.getenv('AWS_REGION', 'us-west-2'),
                create_if_not_exists=False
            )
            retriever = RAGRetriever(vector_store, embeddings)

            # プロジェクト全体のサマリからキーワードを抽出
            all_text = ""
            for i, summary in enumerate(file_summaries[:3]):  # 最初の3ファイルから
                all_text += f"{summary.get('file_name', '')} {summary.get('analysis', '')}[:500] "
                if len(all_text) > 1000:
                    break

            # 類似プロジェクト情報を検索（現在のプロジェクト以外から）
            results = retriever.get_cross_project_insights(
                query=all_text[:1000],
                exclude_project=project_name,
                k=8
            )

            if results:
                rag_context = "\n\n## 過去の類似プロジェクト情報（RAG）\n\n"
                rag_context += "以下は、過去の類似プロジェクトから抽出された知見です。パターン認識や リスク予測に活用してください。\n\n"
                rag_context += retriever.format_context_for_prompt(results, max_chars=8000)
                print(f"[INFO] RAGから{len(results)}件の類似プロジェクト情報を取得")
            else:
                print(f"[INFO] 類似プロジェクトが見つかりませんでした")

        except Exception as e:
            print(f"[WARN] RAG検索でエラーが発生しました: {e}")
            # RAGが失敗しても処理は継続

    # 改善版のプロンプトを使用
    from improved_prompts import get_final_reflection_prompt

    # テンプレート読み込み
    template_path = Path(__file__).parent / "案件情報ノート.md"
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    else:
        template = "リフレクションノートのテンプレートが見つかりません。"

    # ファイル分析結果を統合
    summaries_text = "\n\n".join([
        f"=== {summary['file_name']} の分析結果 ===\n{summary['analysis']}"
        for summary in file_summaries
    ])

    # RAGコンテキストを統合
    if rag_context:
        summaries_text = summaries_text + "\n\n" + rag_context

    # 改善版プロンプトを生成
    prompt = get_final_reflection_prompt(
        project_name=project_name,
        template=template,
        summaries_text=summaries_text
    )

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text, summaries_text
    except Exception as e:
        if _is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[ERROR] Gemini API呼び出しエラー: {e}")
            raise


def save_reflection_note(project_name: str, content: str, summaries_text: str = None):
    """リフレクションノートをファイルに保存"""
    output_path = Path(OUTPUT_DIR) / project_name
    output_path.mkdir(parents=True, exist_ok=True)

    dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 使用しているモデル名を取得（環境変数から）
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    # モデル名を簡潔にする
    model_short = model_name.replace('.', '-')

    # 変換モードも含める
    conversion_mode = os.getenv('CONVERSION_MODE', 'cloud_pdf')

    note_file = output_path / f"{dt}_{model_short}_{conversion_mode}_reflection_note.md"
    with open(note_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[INFO] 出力: {note_file}")

    # summaries_textも保存（提供されている場合）
    if summaries_text:
        summaries_file = output_path / f"{dt}_{model_short}_{conversion_mode}_file_summaries.md"
        with open(summaries_file, 'w', encoding='utf-8') as f:
            f.write(f"# 個別ファイル分析結果\n\n{summaries_text}")
        print(f"[INFO] 分析結果詳細: {summaries_file}")


def cleanup_temp_files():
    """一時ファイルをクリーンアップ"""
    temp_dir = Path(TEMP_DIR)
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
        print("[INFO] 一時ファイルをクリーンアップしました")


def process_project(service, client: genai.Client, project: Dict[str, str]) -> bool:
    """1つの案件を処理 (Google Drive Cloud PDF経由でGemini分析)"""
    project_name = project['name']
    project_id = project['id']

    print(f"\n[INFO] === {project_name} の処理開始 ===")

    # 変換モード
    conversion_mode = os.getenv('CONVERSION_MODE', 'cloud_pdf')
    print(f"[INFO] 変換モード: {conversion_mode}")

    try:
        # ファイル一覧取得
        files = list_files_in_folder(service, project_id)
        print(f"[INFO] ファイル取得: {len(files)}件")

        if not files:
            print(f"[WARN] ファイルが見つかりませんでした。")
            return False

        # 各ファイルを処理してGeminiで分析
        file_summaries = []
        for i, file in enumerate(files, 1):
            print(f"\n[INFO] --- ファイル {i}/{len(files)}: {file['name']} ---")

            # Cloud PDFモードの場合
            if conversion_mode == 'cloud_pdf':
                # ファイルをダウンロード・変換
                file_path, converted_mime = download_and_convert_file(service, file)
                if file_path:
                    # Files APIで分析
                    print(f"[INFO] Geminiで分析中...")
                    try:
                        analysis = analyze_file_with_gemini(
                            client,
                            file_path,
                            file['name'],
                            converted_mime,
                            None,
                            use_rag=True,
                            project_name=project_name
                        )
                        file_summaries.append({
                            'file_name': file['name'],
                            'analysis': analysis
                        })
                        print(f"[INFO] ✓ 分析完了")
                    except Exception as e:
                        print(f"[ERROR] 分析失敗: {e}")
                        print(f"[WARN] テキスト抽出にフォールバック")

                        # テキスト抽出フォールバック
                        text = extract_text_from_file(service, file)
                        if text.strip():
                            try:
                                analysis = analyze_file_with_gemini(
                                    client,
                                    None,
                                    file['name'],
                                    None,
                                    text,
                                    use_rag=True,
                                    project_name=project_name
                                )
                                file_summaries.append({
                                    'file_name': file['name'],
                                    'analysis': analysis
                                })
                                print(f"[INFO] ✓ テキスト分析完了")
                            except Exception as e2:
                                print(f"[ERROR] テキスト分析も失敗: {e2}")
                else:
                    print(f"[WARN] ファイル処理失敗、テキスト抽出を試みます")

                    # テキスト抽出フォールバック
                    text = extract_text_from_file(service, file)
                    if text.strip():
                        try:
                            analysis = analyze_file_with_gemini(
                                client,
                                None,
                                file['name'],
                                None,
                                text,
                                use_rag=True,
                                project_name=project_name
                            )
                            file_summaries.append({
                                'file_name': file['name'],
                                'analysis': analysis
                            })
                            print(f"[INFO] ✓ テキスト分析完了")
                        except Exception as e:
                            print(f"[ERROR] 分析失敗: {e}")

            # テキストモードの場合
            else:
                # テキスト抽出
                text = extract_text_from_file(service, file)
                if not text.strip():
                    print(f"[WARN] テキストが抽出できませんでした。スキップします。")
                    continue

                # Geminiで分析
                print(f"[INFO] Geminiで分析中...")
                try:
                    analysis = analyze_file_with_gemini(
                        client,
                        None,
                        file['name'],
                        None,
                        text,
                        use_rag=True,
                        project_name=project_name
                    )
                    file_summaries.append({
                        'file_name': file['name'],
                        'analysis': analysis
                    })
                    print(f"[INFO] ✓ 分析完了")
                except Exception as e:
                    print(f"[ERROR] 分析失敗: {e}")
                    print(f"[WARN] このファイルをスキップして続行します")
                    continue

        if not file_summaries:
            print(f"[WARN] 分析可能なファイルがありませんでした。")
            return False

        # 最終的なリフレクションノート生成
        print(f"\n[INFO] === 最終的なリフレクションノート生成中 ===")
        print(f"[INFO] 統合するファイル数: {len(file_summaries)}件")
        reflection_note, summaries_text = generate_final_reflection_note(client, project_name, file_summaries)

        # 保存
        save_reflection_note(project_name, reflection_note, summaries_text)

        print(f"[INFO] === {project_name} の処理完了 ===")
        return True

    except Exception as e:
        print(f"[ERROR] {project_name} の処理中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン処理"""
    print("[INFO] LISA PoC - リフレクションノート自動生成 (Google Drive Cloud PDF版)")
    print("[INFO] 外部バイナリ不要・クラウドPDF変換対応")
    print()

    # 変換モード確認
    conversion_mode = os.getenv('CONVERSION_MODE', 'cloud_pdf')
    print(f"[INFO] 変換モード: {conversion_mode}")
    if conversion_mode == 'cloud_pdf':
        print("[INFO] Google Drive APIを使用してクラウド上でPDF変換します")
    else:
        print("[INFO] テキスト抽出モードで動作します")

    # 環境変数チェック
    api_key = os.getenv('GEMINI_API_KEY')
    projects_folder_id = os.getenv('PROJECTS_FOLDER_ID')
    project_names = os.getenv('PROJECT_NAMES', '*')

    if not api_key:
        print("[ERROR] GEMINI_API_KEY が設定されていません。")
        print(".envファイルを確認してください。")
        sys.exit(1)

    if not projects_folder_id:
        print("[ERROR] PROJECTS_FOLDER_ID が設定されていません。")
        print(".envファイルを確認してください。")
        sys.exit(1)

    # OAuth認証（Google Drive用）
    print("[INFO] OAuth認証中...")
    creds = authenticate()
    print("[INFO] OAuth認証完了")

    # Google Driveサービス取得
    service = get_drive_service(creds)

    # Geminiクライアント初期化
    print("[INFO] Gemini APIクライアント初期化中...")
    gemini_client = initialize_gemini_client()
    print("[INFO] Gemini APIクライアント初期化完了")

    # 案件フォルダ一覧取得
    print(f"[INFO] 案件情報フォルダ: /案件情報/ (ID: {projects_folder_id})")
    all_projects = list_project_folders(service, projects_folder_id)

    # フィルタリング
    target_projects = filter_projects(all_projects, project_names)

    if not target_projects:
        print("[WARN] 処理対象の案件が見つかりませんでした。")
        sys.exit(0)

    print(f"[INFO] 処理対象案件: {', '.join([p['name'] for p in target_projects])} ({len(target_projects)}件)")

    # 各案件を処理
    success_count = 0
    fail_count = 0

    try:
        for project in target_projects:
            if process_project(service, gemini_client, project):
                success_count += 1
            else:
                fail_count += 1
    finally:
        # 一時ファイルのクリーンアップ
        # cleanup_temp_files()
        pass

    # サマリー出力
    print()
    print("=" * 40)
    print("処理サマリー")
    print("=" * 40)
    print(f"成功: {success_count}件")
    print(f"失敗: {fail_count}件")
    print("=" * 40)


if __name__ == "__main__":
    main()
