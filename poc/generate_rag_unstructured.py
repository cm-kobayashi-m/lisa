#!/usr/bin/env python3
"""
LISA PoC - 最適化版RAG用ベクトルDB構築スクリプト

Phase 1改善:
- PDF戦略の事前判定（hi_res廃止）
- バッチ処理（埋め込み・S3保存）
- キャッシュ機構

使用方法:
    # すべてのプロジェクトを処理
    python3 generate_rag_unstructured_optimized.py

    # 特定のプロジェクトのみ処理
    python3 generate_rag_unstructured_optimized.py --project "LISAのPoCテスト"

    # キャッシュをクリア
    python3 generate_rag_unstructured_optimized.py --clear-cache
"""

import os
import sys
import yaml
import json
import argparse
import gc
import traceback
import hashlib
import tempfile
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
from pathlib import Path

import fitz  # PyMuPDF

# Google Drive API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# LangChain（画像OCR用）
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Unstructured ライブラリ
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title

# テキスト分割用
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RAGモジュール
from rag.vector_store import S3VectorStore, Document
from rag.embeddings import GeminiEmbeddings
from rag.document_classifier import DocumentClassifier

# プロジェクト設定
from project_config import ProjectConfig

# 定数
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "token.yaml"
CREDENTIALS_FILE = "credentials.json"
TEMP_DIR = "temp_files"
MAX_FILE_SIZE = 30  # MB（PDF/Officeファイル用）
MAX_WORKERS = 10  # Phase 1: ダウンロード並列度を上げる
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', "gemini-embedding-001")
DIMENSION = int(os.getenv('DIMENSION', 1536))

# バッチ処理設定
EMBEDDING_BATCH_SIZE = 50  # 埋め込みのバッチサイズ
S3_BATCH_SIZE = 50  # S3保存のバッチサイズ

# キャッシュ設定
CACHE_DIR = ".rag_cache"
CACHE_VERSION = "v1"  # キャッシュバージョン（戦略変更時に更新）

# MS Office MIMEタイプ定義
OFFICE_WORD_MIMES = [
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
]
OFFICE_EXCEL_MIMES = [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
]
OFFICE_PPT_MIMES = [
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint",
]

# RAG設定
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME", "lisa-poc-vectors")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "project-documents")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# 環境変数読み込み
load_dotenv()

# 共通モジュールのインポート
from utils.llm_response import extract_content as _extract_content

# ファイル処理のタイムアウト設定（秒）
PARTITION_TIMEOUT = int(os.getenv("PARTITION_TIMEOUT", 60*10))  # デフォルト10分


class FileCache:
    """処理済みファイルのキャッシュ管理"""

    def __init__(self):
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / f"processed_files_{CACHE_VERSION}.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """キャッシュファイルを読み込み"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """キャッシュファイルを保存（日本語を正しく表示）"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def get_file_hash(self, file_id: str, modified_time: str) -> str:
        """ファイルのハッシュ値を生成"""
        hash_input = f"{file_id}:{modified_time}:{CACHE_VERSION}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def is_processed(self, file_id: str, modified_time: str) -> bool:
        """ファイルが処理済みか確認"""
        file_hash = self.get_file_hash(file_id, modified_time)
        return file_hash in self.cache

    def mark_processed(self, file_id: str, modified_time: str, chunk_count: int,
                      project_name: str = None, file_name: str = None,
                      document_type: str = None):
        """ファイルを処理済みとしてマーク（詳細メタデータ付き）"""
        file_hash = self.get_file_hash(file_id, modified_time)
        self.cache[file_hash] = {
            "file_id": file_id,
            "file_name": file_name,
            "project_name": project_name,
            "modified_time": modified_time,
            "processed_at": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "document_type": document_type,
            "cache_version": CACHE_VERSION
        }
        self._save_cache()

    def clear_cache(self):
        """キャッシュをクリア"""
        self.cache = {}
        self._save_cache()
        print("[INFO] キャッシュをクリアしました")


def authenticate():
    """OAuth 2.0認証（Google Drive API用）"""
    creds = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as token:
            token_data = yaml.safe_load(token)
            creds = Credentials(
                token=token_data["token"],
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri"),
                client_id=token_data.get("client_id"),
                client_secret=token_data.get("client_secret"),
                scopes=token_data.get("scopes"),
            )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"[ERROR] {CREDENTIALS_FILE} が見つかりません。")
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        token_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
        }
        with open(TOKEN_FILE, "w") as token:
            yaml.safe_dump(token_data, token, default_flow_style=False)

    return creds


def get_drive_service(creds):
    """Google Drive サービス取得"""
    return build("drive", "v3", credentials=creds)


def list_project_folders(service, projects_folder_id: str) -> List[Dict[str, str]]:
    """案件情報フォルダ配下のフォルダ一覧を取得"""
    query = f"'{projects_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"

    results = (
        service.files()
        .list(
            q=query,
            fields="files(id, name)",
            orderBy="name",
            supportsAllDrives=True,  # 共有ドライブサポート
            includeItemsFromAllDrives=True,  # 共有ドライブのアイテムを含める
        )
        .execute()
    )

    return results.get("files", [])


def list_files_in_folder(service, folder_id: str) -> List[Dict[str, str]]:
    """フォルダ配下のファイル一覧を再帰的に取得"""
    all_files = []

    # 処理対象のMIMEタイプ
    target_mime_types\
        = [
        "application/pdf",
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.google-apps.presentation",
        # MS Office形式
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        # テキスト・Markdown
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        # 画像（OCRなしの場合はスキップされる）
        "image/png",
        "image/jpeg",
        "image/jpg",
    ]

    def _list_recursive(current_folder_id: str):
        """再帰的にファイルとフォルダを取得"""
        query = f"'{current_folder_id}' in parents and trashed=false"

        results = (
            service.files()
            .list(
                q=query,
                fields="files(id, name, mimeType, size, createdTime, modifiedTime)",
                pageSize=1000,
                supportsAllDrives=True,  # 共有ドライブサポート
                includeItemsFromAllDrives=True,  # 共有ドライブのアイテムを含める
            )
            .execute()
        )

        items = results.get("files", [])

        for item in items:
            if item["mimeType"] == "application/vnd.google-apps.folder":
                _list_recursive(item["id"])
            elif item["mimeType"] in target_mime_types:
                all_files.append(item)

    _list_recursive(folder_id)
    return all_files


class OptimizedChunker:
    """最適化されたチャンク化処理（Phase 1改善）"""

    def __init__(
        self,
        embeddings: GeminiEmbeddings,
        file_cache: FileCache,
        classifier: DocumentClassifier
    ):
        self.embeddings = embeddings
        self.file_cache = file_cache
        self.classifier = classifier
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        # チャンク処理時の状態管理
        self.current_document_type = None
        self.current_document_confidence = 0.0

    def check_pdf_has_text(self, file_path: str) -> bool:
        """PDFにテキストが含まれているか事前チェック（Phase 1改善）"""

        try:
            doc = fitz.open(file_path)
            total_text_length = 0
            for page in doc:
                text = page.get_text()
                total_text_length += len(text)
                if total_text_length > 100:  # テキストがある程度あれば
                    doc.close()
                    return True
            doc.close()
            return False
        except Exception as e:
            print(f"    [WARN] PDF事前チェックエラー: {e}")
            return False

    def chunk_file_with_unstructured(
        self, service, file_info: Dict, project_name: str
    ) -> List[Dict]:
        """
        unstructuredライブラリでファイルを読み込み、チャンク化する（最適化版）

        Returns:
            チャンク情報のリスト（text, metadata）
        """
        file_id = file_info["id"]
        file_name = file_info["name"]
        mime_type = file_info["mimeType"]
        modified_time = file_info.get("modifiedTime", "")

        # ファイル名プレフィックスをログに追加
        log_prefix = f"[{file_name}]"

        print("\n  ========================================")
        print(f"  {log_prefix} 処理中")
        print("  ========================================")

        # キャッシュチェック（Phase 1改善）
        if self.file_cache.is_processed(file_id, modified_time):
            print(f"    {log_prefix} [SKIP] 処理済み（キャッシュヒット）")
            return []

        # ファイルサイズチェック
        file_size = int(file_info.get("size", 0))
        print(
            f"    {log_prefix} [情報] ファイルサイズ: {file_size / 1024:.1f}KB, MIMEタイプ: {mime_type}"
        )

        if file_size > MAX_FILE_SIZE * 1024 * 1024:
            print(
                f"    {log_prefix} [SKIP] ファイルサイズが大きすぎます: {file_size / 1024 / 1024:.1f}MB"
            )
            return []

        try:
            # ファイルをダウンロード（一時ファイルへ）
            print(f"    {log_prefix} [開始] ダウンロード処理を開始...")
            tmp_path = self._download_to_tempfile(
                service, file_id, file_name, mime_type
            )

            if not tmp_path:
                print(f"    {log_prefix} [SKIP] ファイルを読み込めませんでした")
                return []

            # unstructuredでElements抽出（タイムアウト付き）
            print(f"    {log_prefix} [処理] ファイルを解析中... (タイムアウト: {PARTITION_TIMEOUT}秒)")

            try:
                # func_timeoutでタイムアウト付き実行
                elements = func_timeout(
                    PARTITION_TIMEOUT,
                    self._partition_elements_optimized,
                    args=(tmp_path, mime_type, file_name)
                )
            except FunctionTimedOut:
                print(f"    {log_prefix} [WARN] ファイルの処理がタイムアウトしました ({PARTITION_TIMEOUT}秒)")
                print(f"    {log_prefix}        処理をスキップして次のファイルを処理します")
                elements = []
            except Exception as e:
                print(f"    {log_prefix} [ERROR] ファイル処理エラー: {e}")
                elements = []

            # 一時ファイル削除
            try:
                os.unlink(tmp_path)
            except:
                pass

            if not elements:
                print(f"    {log_prefix} [SKIP] コンテンツを抽出できませんでした")
                return []

            # 構造的なチャンク化
            chunks = self._elements_to_chunks(elements, project_name, file_name)
            # ファイルのメタデータを追加
            for chunk in chunks:
                if "modifiedTime" in file_info:
                    chunk["metadata"]["modified_at"] = file_info["modifiedTime"]
                if "createdTime" in file_info:
                    chunk["metadata"]["created_at"] = file_info["createdTime"]
                # データソース情報を追加
                if "data_source" in file_info:
                    chunk["metadata"]["data_source"] = file_info["data_source"]
                if "folder_id" in file_info:
                    chunk["metadata"]["source_id"] = file_info["folder_id"]

            print(f"    {log_prefix} {len(chunks)}個のチャンクを生成")

            # 注: mark_processedはS3保存後にprocess_batchで呼び出される

            return chunks

        except Exception as e:
            print(f"    {log_prefix} [ERROR] 処理エラー: {e}")
            print(f"    {log_prefix} [DEBUG] エラー詳細: {traceback.format_exc()}")
            return []

    def _download_to_tempfile(
        self, service, file_id: str, file_name: str, mime_type: str
    ) -> Optional[str]:
        """ファイルを一時ファイルにダウンロード（Phase 1改善）"""
        log_prefix = f"[{file_name}]"
        try:
            # 一時ファイル作成
            suffix = os.path.splitext(file_name)[1] or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name

                # Google Docs → DOCX
                if mime_type == "application/vnd.google-apps.document":
                    print(f"    {log_prefix} [ダウンロード] Google DocsをDOCX形式でエクスポート中...")
                    request = service.files().export_media(
                        fileId=file_id,
                        mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                # Google Sheets → XLSX
                elif mime_type == "application/vnd.google-apps.spreadsheet":
                    print(f"    {log_prefix} [ダウンロード] Google SheetsをXLSX形式でエクスポート中...")
                    request = service.files().export_media(
                        fileId=file_id,
                        mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                # Google Slides → PPTX
                elif mime_type == "application/vnd.google-apps.presentation":
                    print(f"    {log_prefix} [ダウンロード] Google SlidesをPPTX形式でエクスポート中...")
                    request = service.files().export_media(
                        fileId=file_id,
                        mimeType="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                # その他のファイル
                else:
                    print(f"    {log_prefix} [ダウンロード] ファイルダウンロード中...")
                    request = service.files().get_media(
                        fileId=file_id,
                        supportsAllDrives=True  # 共有ドライブサポート
                    )

                # ダウンロード実行（直接ファイルに書き込み）
                downloader = MediaIoBaseDownload(tmp, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

                file_size = os.path.getsize(tmp_path)
                print(f"    {log_prefix} [ダウンロード] 完了 ({file_size / 1024 / 1024:.2f}MB)")

                return tmp_path

        except Exception as e:
            print(f"    {log_prefix} [ERROR] ダウンロードエラー: {e}")
            return None

    def _partition_elements_optimized(
        self, file_path: str, mime_type: str, file_name: str
    ) -> List[Any]:
        """ファイル形式に応じたpartition処理（最適化版）"""
        log_prefix = f"[{file_name}]"
        try:
            # PDF（Phase 1: 戦略最適化）
            if mime_type == "application/pdf":
                # テキスト有無を事前チェック
                has_text = self.check_pdf_has_text(file_path)

                if has_text:
                    print(f"    {log_prefix} [処理] PDFを解析中 (fast戦略 - テキストあり)...")
                    # テキストがあるPDFは高速処理
                    elements = partition_pdf(
                        filename=file_path,
                        strategy="fast",
                        extract_images_in_pdf=False,  # 画像抽出無効
                        infer_table_structure=False,  # テーブル推論無効
                        metadata_filename=file_name
                    )
                else:
                    print(f"    {log_prefix} [処理] PDFを解析中 (auto戦略 - スキャンPDF疑い)...")
                    # スキャンPDFの可能性がある場合
                    elements = partition_pdf(
                        filename=file_path,
                        strategy="auto",
                        infer_table_structure=True,
                        extract_images_in_pdf=False,
                        metadata_filename=file_name
                    )
                return elements

            # MS Word / Google Docs (DOCX形式)
            elif mime_type in OFFICE_WORD_MIMES or mime_type == "application/vnd.google-apps.document":
                print(f"    {log_prefix} [処理] Word/Docsドキュメントを解析中...")
                return partition_docx(filename=file_path, metadata_filename=file_name)

            # MS Excel / Google Sheets (XLSX形式)
            elif mime_type in OFFICE_EXCEL_MIMES or mime_type == "application/vnd.google-apps.spreadsheet":
                print(f"    {log_prefix} [処理] Excel/Sheetsスプレッドシートを解析中...")
                return partition_xlsx(filename=file_path, metadata_filename=file_name)

            # MS PowerPoint / Google Slides (PPTX形式)
            elif mime_type in OFFICE_PPT_MIMES or mime_type == "application/vnd.google-apps.presentation":
                print(f"    {log_prefix} [処理] PowerPoint/Slidesプレゼンテーションを解析中...")
                return partition_pptx(filename=file_path, metadata_filename=file_name)

            # Markdown
            elif mime_type in ["text/markdown", "text/x-markdown"]:
                print(f"    {log_prefix} [処理] Markdownドキュメントを解析中...")
                return partition_md(filename=file_path, metadata_filename=file_name)

            # プレーンテキスト
            elif mime_type.startswith("text/"):
                print(f"    {log_prefix} [処理] テキストドキュメントを解析中...")
                return partition_text(filename=file_path, metadata_filename=file_name)

            # 画像（Gemini LLMでOCR処理）
            elif mime_type in ["image/png", "image/jpeg", "image/jpg"]:
                print(f"    {log_prefix} [処理] 画像ファイルをGemini LLMでOCR処理中...")
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                return self._process_image_with_gemini(file_bytes, mime_type, file_name)

            else:
                print(f"    {log_prefix} [WARN] 未対応のMIMEタイプ: {mime_type}")
                return []

        except Exception as e:
            print(f"    {log_prefix} [ERROR] Elements抽出エラー: {e}")
            print(f"    {log_prefix} [DEBUG] {traceback.format_exc()}")
            return []

    def _process_image_with_gemini(
        self, file_bytes: bytes, mime_type: str, file_name: str
    ) -> List[Any]:
        """LangChain経由でGemini LLMを使用して画像からテキストを抽出（OCR）"""
        try:
            # API キーの確認
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print(
                    "    [WARN] GEMINI_API_KEY が設定されていないため、画像処理をスキップ"
                )
                return []

            # LangChainのGeminiモデル初期化
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro ")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1,  # 低めの温度で正確性重視
                max_output_tokens=32768,
            )

            # OCRプロンプト
            ocr_prompt = """
この画像からすべてのテキストを抽出してください。
以下の形式で出力してください：

1. 見出しやタイトルがあれば、それらを明確に識別してください
2. 本文テキストを段落ごとに分けてください
3. 表やリストがあれば、その構造を保持してください
4. 日本語・英語両方のテキストを正確に抽出してください

テキストのみを出力し、画像の説明や解釈は不要です。
"""

            # 画像をbase64エンコード
            import base64
            image_data = base64.b64encode(file_bytes).decode()

            # LangChainメッセージ形式で画像とプロンプトを送信
            message = HumanMessage(
                content=[
                    {"type": "text", "text": ocr_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    }
                ]
            )

            # LLMでOCR実行
            response = llm.invoke([message])

            if response and response.content:
                # unstructuredのText要素として返す
                from unstructured.documents.elements import Text

                # 段落ごとに分割してElement化
                paragraphs = _extract_content(response).split("\n\n")
                elements = []
                for para in paragraphs:
                    if para.strip():
                        # unstructuredのTextエレメントを使用
                        elements.append(Text(text=para.strip()))

                print(f"    [処理] LangChain Gemini OCRで{len(elements)}個の要素を抽出")
                return elements
            else:
                print("    [WARN] LangChain Geminiからのレスポンスが空です")
                return []

        except Exception as e:
            print(f"    [ERROR] LangChain Gemini OCR処理エラー: {e}")
            print(f"    [DEBUG] {traceback.format_exc()}")
            return []

    def _elements_to_chunks(
        self, elements: List[Any], project_name: str, file_name: str
    ) -> List[Dict]:
        """Elementsを構造的にチャンク化（改善版）"""
        log_prefix = f"[{file_name}]"
        try:
            print(f"    {log_prefix} [処理] {len(elements)}個のElementsをチャンク化中...")

            # ドキュメント分類（1ファイルにつき1回のLLM呼び出し）
            print(f"    {log_prefix} [分類] ドキュメント種別を判定中...")
            document_type, confidence = self.classifier.classify_from_elements(
                elements=elements,
                file_name=file_name,
                max_elements=30  # 先頭30要素から判定
            )
            print(f"    {log_prefix} [分類] 判定結果: {document_type} (信頼度: {confidence:.2f})")

            # 状態を保存
            self.current_document_type = document_type
            self.current_document_confidence = confidence

            # ドキュメント種別に応じた処理を選択
            if document_type == "technical_document":
                return self._chunk_technical_document(elements, project_name, file_name)
            elif document_type == "meeting_minutes":
                return self._chunk_meeting_minutes(elements, project_name, file_name)
            elif document_type == "proposal":
                return self._chunk_proposal(elements, project_name, file_name)
            else:
                # デフォルト: 適応的なチャンク化
                return self._chunk_by_title_adaptive(elements, project_name, file_name,
                                                    document_type, confidence)

        except Exception as e:
            log_prefix = f"[{file_name}]"
            print(f"    {log_prefix} [ERROR] チャンク化エラー: {e}")
            print(f"    {log_prefix} [DEBUG] {traceback.format_exc()}")
            # フォールバック：単純な分割
            return self._simple_chunk(elements, project_name, file_name)

    def _chunk_by_title_adaptive(
        self, elements: List[Any], project_name: str, file_name: str,
        document_type: str, confidence: float
    ) -> List[Dict]:
        """適応的なタイトルベースチャンク化"""
        log_prefix = f"[{file_name}]"

        # 文書サイズに応じてパラメータを調整
        total_text_length = sum(len(getattr(e, "text", "")) for e in elements)

        if total_text_length > 50000:  # 長文書
            max_chars = 1500
            new_after = 1200
        elif total_text_length > 10000:  # 中文書
            max_chars = 1200
            new_after = 1000
        else:  # 短文書
            max_chars = 800
            new_after = 600

        print(f"    {log_prefix} [適応] 文書サイズ: {total_text_length}文字, max_chars: {max_chars}")

        # chunk_by_titleで論理単位にグループ化
        title_chunks = chunk_by_title(
            elements,
            max_characters=max_chars,
            new_after_n_chars=new_after,
            combine_text_under_n_chars=200,
            multipage_sections=True,
        )

        chunks = []
        for idx, tc in enumerate(title_chunks):
            # テキストを抽出
            text = self._extract_text_from_chunk(tc)
            if not text:
                continue

            # 長すぎる場合は段落境界で分割（改善版）
            if len(text) > max_chars * 1.5:
                sub_texts = self._split_by_paragraph(text, max_chars)
                for sub_idx, sub_text in enumerate(sub_texts):
                    chunk_data = self._create_enhanced_chunk(
                        sub_text, tc, project_name, file_name,
                        idx, sub_idx, document_type, confidence
                    )
                    chunks.append(chunk_data)
            else:
                # そのまま使用（純度維持）
                chunk_data = self._create_enhanced_chunk(
                    text, tc, project_name, file_name,
                    idx, 0, document_type, confidence
                )
                chunks.append(chunk_data)

        # チャンク間の関係性を追加
        self._add_chunk_relationships(chunks)

        return chunks

    def _chunk_technical_document(
        self, elements: List[Any], project_name: str, file_name: str
    ) -> List[Dict]:
        """技術文書向けチャンク化（セクション構造重視）"""
        log_prefix = f"[{file_name}]"
        print(f"    {log_prefix} [技術文書] セクション構造を重視したチャンク化")

        # セクション構造を保持しながらチャンク化
        title_chunks = chunk_by_title(
            elements,
            max_characters=1500,  # 技術文書は長めに
            new_after_n_chars=1300,
            combine_text_under_n_chars=300,
            multipage_sections=True,
        )

        chunks = []
        for idx, tc in enumerate(title_chunks):
            text = self._extract_text_from_chunk(tc)
            if not text:
                continue

            # コードブロックを検出して保持
            has_code = self._detect_code_block(text)

            # 技術文書は段落境界を尊重
            if len(text) > 2000:
                sub_texts = self._split_by_paragraph(text, 1500)
                for sub_idx, sub_text in enumerate(sub_texts):
                    chunk_data = self._create_enhanced_chunk(
                        sub_text, tc, project_name, file_name,
                        idx, sub_idx, "technical_document", self.current_document_confidence
                    )
                    chunk_data["metadata"]["has_code"] = has_code
                    chunks.append(chunk_data)
            else:
                chunk_data = self._create_enhanced_chunk(
                    text, tc, project_name, file_name,
                    idx, 0, "technical_document", self.current_document_confidence
                )
                chunk_data["metadata"]["has_code"] = has_code
                chunks.append(chunk_data)

        self._add_chunk_relationships(chunks)
        return chunks

    def _chunk_meeting_minutes(
        self, elements: List[Any], project_name: str, file_name: str
    ) -> List[Dict]:
        """議事録向けチャンク化（時系列と発言者重視）"""
        log_prefix = f"[{file_name}]"
        print(f"    {log_prefix} [議事録] 時系列と発言者を重視したチャンク化")

        # 議事録は短めのチャンクで時系列を保持
        title_chunks = chunk_by_title(
            elements,
            max_characters=800,
            new_after_n_chars=600,
            combine_text_under_n_chars=150,
            multipage_sections=True,
        )

        chunks = []
        for idx, tc in enumerate(title_chunks):
            text = self._extract_text_from_chunk(tc)
            if not text:
                continue

            # 発言者を検出
            speakers = self._detect_speakers(text)

            chunk_data = self._create_enhanced_chunk(
                text, tc, project_name, file_name,
                idx, 0, "meeting_minutes", self.current_document_confidence
            )
            chunk_data["metadata"]["speakers"] = speakers
            chunk_data["metadata"]["temporal_index"] = idx  # 時系列インデックス
            chunks.append(chunk_data)

        self._add_chunk_relationships(chunks)
        return chunks

    def _chunk_proposal(
        self, elements: List[Any], project_name: str, file_name: str
    ) -> List[Dict]:
        """提案書向けチャンク化（章立て構造重視）"""
        log_prefix = f"[{file_name}]"
        print(f"    {log_prefix} [提案書] 章立て構造を重視したチャンク化")

        # 提案書は標準的なサイズでセクションを保持
        title_chunks = chunk_by_title(
            elements,
            max_characters=1200,
            new_after_n_chars=1000,
            combine_text_under_n_chars=200,
            multipage_sections=True,
        )

        chunks = []
        for idx, tc in enumerate(title_chunks):
            text = self._extract_text_from_chunk(tc)
            if not text:
                continue

            # 数値データ（金額、期間等）を検出
            has_numbers = self._detect_numbers(text)

            if len(text) > 1400:
                sub_texts = self._split_by_paragraph(text, 1200)
                for sub_idx, sub_text in enumerate(sub_texts):
                    chunk_data = self._create_enhanced_chunk(
                        sub_text, tc, project_name, file_name,
                        idx, sub_idx, "proposal", self.current_document_confidence
                    )
                    chunk_data["metadata"]["has_numbers"] = has_numbers
                    chunks.append(chunk_data)
            else:
                chunk_data = self._create_enhanced_chunk(
                    text, tc, project_name, file_name,
                    idx, 0, "proposal", self.current_document_confidence
                )
                chunk_data["metadata"]["has_numbers"] = has_numbers
                chunks.append(chunk_data)

        self._add_chunk_relationships(chunks)
        return chunks

    def _simple_chunk(
        self, elements: List[Any], project_name: str, file_name: str
    ) -> List[Dict]:
        """フォールバック用の単純なチャンク分割"""
        log_prefix = f"[{file_name}]"
        try:
            # ドキュメント分類（フォールバック時も実行）
            print(f"    {log_prefix} [分類] ドキュメント種別を判定中（フォールバック）...")
            document_type, confidence = self.classifier.classify_from_elements(
                elements=elements,
                file_name=file_name,
                max_elements=10
            )
            print(f"    {log_prefix} [分類] 判定結果: {document_type} (信頼度: {confidence:.2f})")

            # Elementsからテキストを抽出
            all_text = []
            for element in elements:
                text = getattr(element, "text", "") or ""
                if text:
                    all_text.append(text)

            full_text = "\n\n".join(all_text)

            if not full_text:
                return []

            # テキスト分割
            chunks = []
            text_chunks = self.text_splitter.split_text(full_text)

            for idx, chunk_text in enumerate(text_chunks):
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "project_name": project_name,
                            "file_name": file_name,
                            "chunk_index": idx,
                            "title": f"Chunk {idx + 1}",
                            # ドキュメント種別
                            "document_type": document_type,
                            "document_type_confidence": confidence,
                            "element_types": [],
                            "pages": [],
                            "importance": "medium",
                        },
                    }
                )

            return chunks
        except Exception as e:
            print(f"    {log_prefix} [ERROR] フォールバックチャンク化エラー: {e}")
            return []

    # ========== ヘルパーメソッド（改善版） ==========

    def _extract_text_from_chunk(self, chunk: Any) -> str:
        """チャンクからテキストを抽出"""
        text_parts = []
        elements = chunk.elements if hasattr(chunk, "elements") else [chunk]

        for element in elements:
            elem_text = getattr(element, "text", "") or ""
            if elem_text:
                text_parts.append(elem_text)

        return " ".join(text_parts).strip()

    def _split_by_paragraph(self, text: str, max_length: int) -> List[str]:
        """段落境界でテキストを分割（文脈保持）"""
        # 段落で分割
        paragraphs = text.split('\n\n')

        result = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 現在のチャンクに追加しても最大長を超えない場合
            if len(current_chunk) + len(para) + 2 < max_length:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 現在のチャンクを保存して新しいチャンク開始
                if current_chunk:
                    result.append(current_chunk)

                # 段落自体が長すぎる場合は、文境界で分割
                if len(para) > max_length:
                    sentences = self._split_by_sentence(para)
                    temp_chunk = ""
                    for sent in sentences:
                        if len(temp_chunk) + len(sent) + 1 < max_length:
                            temp_chunk = temp_chunk + " " + sent if temp_chunk else sent
                        else:
                            if temp_chunk:
                                result.append(temp_chunk)
                            temp_chunk = sent
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = para

        # 最後のチャンクを追加
        if current_chunk:
            result.append(current_chunk)

        return result if result else [text[:max_length]]

    def _split_by_sentence(self, text: str) -> List[str]:
        """文境界でテキストを分割"""
        import re
        # 日本語と英語の文境界で分割
        sentences = re.split(r'(?<=[。！？\.!?])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_enhanced_chunk(
        self, text: str, element_chunk: Any, project_name: str, file_name: str,
        section_idx: int, sub_idx: int, document_type: str, confidence: float
    ) -> Dict:
        """強化されたメタデータを持つチャンク作成"""

        # 基本メタデータ
        metadata = {
            "project_name": project_name,
            "file_name": file_name,
            "chunk_index": section_idx * 100 + sub_idx,  # 階層的インデックス
            "document_type": document_type,
            "document_type_confidence": confidence,
        }

        # タイトル情報
        title = self._extract_title(element_chunk)
        metadata["title"] = title if title else f"Section {section_idx + 1}"

        # 階層情報
        metadata["heading_level"] = self._extract_heading_level(element_chunk)
        metadata["section_path"] = self._build_section_path(element_chunk, section_idx)

        # 構造情報
        metadata["has_table"] = self._contains_table(element_chunk)
        metadata["has_list"] = self._contains_list(element_chunk)

        # Element種別
        metadata["element_types"] = self._extract_element_types(element_chunk)

        # ページ情報
        metadata["pages"] = self._extract_pages(element_chunk)

        # 位置情報
        metadata["chunk_position"] = self._determine_position(section_idx)

        # 情報密度
        metadata["information_density"] = self._calculate_information_density(text)

        # 重要度
        metadata["importance"] = self._calculate_importance(text, metadata)

        return {
            "text": text,
            "metadata": metadata
        }

    def _extract_title(self, chunk: Any) -> str:
        """チャンクからタイトルを抽出"""
        if hasattr(chunk, "title"):
            return chunk.title
        elif hasattr(chunk, "metadata") and hasattr(chunk.metadata, "title"):
            return chunk.metadata.title
        return ""

    def _extract_heading_level(self, chunk: Any) -> int:
        """見出しレベルを抽出（1-6）"""
        # Unstructuredの要素タイプから推定
        if hasattr(chunk, "category"):
            category = chunk.category.lower() if isinstance(chunk.category, str) else ""
            if "title" in category:
                # タイトルのサイズやスタイルから推定（簡易版）
                if hasattr(chunk, "metadata"):
                    # フォントサイズなどから推定する場合
                    return 1  # デフォルトで1
            elif "header" in category:
                return 2
            elif "subheader" in category:
                return 3
        return 0  # 見出しではない

    def _build_section_path(self, chunk: Any, section_idx: int) -> str:
        """セクション階層パスを構築"""
        # 簡易版：実際はドキュメント全体の構造から構築
        title = self._extract_title(chunk)
        if title:
            return title
        return f"Section {section_idx + 1}"

    def _contains_table(self, chunk: Any) -> bool:
        """テーブルが含まれるかチェック"""
        if hasattr(chunk, "elements"):
            for elem in chunk.elements:
                if hasattr(elem, "category") and "table" in str(elem.category).lower():
                    return True
        elif hasattr(chunk, "category") and "table" in str(chunk.category).lower():
            return True

        # テキスト内のテーブル記号をチェック
        text = self._extract_text_from_chunk(chunk)
        return "|\t" in text or " | " in text

    def _contains_list(self, chunk: Any) -> bool:
        """リストが含まれるかチェック"""
        if hasattr(chunk, "elements"):
            for elem in chunk.elements:
                if hasattr(elem, "category") and "list" in str(elem.category).lower():
                    return True

        # テキスト内のリスト記号をチェック
        text = self._extract_text_from_chunk(chunk)
        list_patterns = ["\n- ", "\n* ", "\n• ", "\n1. ", "\n2. ", "・"]
        return any(pattern in text for pattern in list_patterns)

    def _extract_element_types(self, chunk: Any) -> List[str]:
        """Element種別を抽出"""
        types = set()
        if hasattr(chunk, "category"):
            types.add(str(chunk.category))
        if hasattr(chunk, "elements"):
            for elem in chunk.elements:
                if hasattr(elem, "category"):
                    types.add(str(elem.category))
        return list(types)

    def _extract_pages(self, chunk: Any) -> List[int]:
        """ページ番号を抽出"""
        pages = set()
        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "page_number"):
            pages.add(chunk.metadata.page_number)
        if hasattr(chunk, "elements"):
            for elem in chunk.elements:
                if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                    pages.add(elem.metadata.page_number)
        return sorted(list(pages))

    def _determine_position(self, section_idx: int) -> str:
        """文書内での位置を判定"""
        # 簡易版：実際の実装では総セクション数を考慮
        if section_idx == 0:
            return "beginning"
        elif section_idx < 3:
            return "beginning"
        else:
            return "middle"

    def _calculate_information_density(self, text: str) -> float:
        """情報密度を計算（0.0-1.0）"""
        if not text:
            return 0.0

        # 簡易版：文字数、数値、専門用語の密度から計算
        text_length = len(text)

        # 数値の密度
        import re
        numbers = re.findall(r'\d+', text)
        number_density = len(numbers) / max(text_length / 100, 1)

        # 句読点の密度（文の複雑さの指標）
        punctuation_count = text.count('。') + text.count('.') + text.count('、') + text.count(',')
        punctuation_density = punctuation_count / max(text_length / 100, 1)

        # カタカナ語（専門用語の指標）
        katakana_words = re.findall(r'[ア-ン]+', text)
        katakana_density = len(katakana_words) / max(text_length / 100, 1)

        # 総合密度
        density = min(1.0, (number_density * 0.3 + punctuation_density * 0.3 + katakana_density * 0.4) / 3)
        return round(density, 2)

    def _calculate_importance(self, text: str, metadata: Dict) -> str:
        """チャンクの重要度を計算"""
        score = 0

        # 見出しレベルによる重み
        heading_level = metadata.get("heading_level", 0)
        if heading_level > 0:
            score += (7 - heading_level) * 10  # レベル1が最も重要

        # 情報密度による重み
        density = metadata.get("information_density", 0)
        score += density * 30

        # 構造要素による重み
        if metadata.get("has_table"):
            score += 20
        if metadata.get("has_list"):
            score += 10

        # 位置による重み
        position = metadata.get("chunk_position", "middle")
        if position == "beginning":
            score += 15

        # スコアから重要度を決定
        if score >= 60:
            return "high"
        elif score >= 30:
            return "medium"
        else:
            return "low"

    def _add_chunk_relationships(self, chunks: List[Dict]) -> None:
        """チャンク間の関係性情報を追加"""
        for i, chunk in enumerate(chunks):
            # 前のセクション
            if i > 0:
                chunk["metadata"]["prev_section"] = chunks[i-1]["metadata"]["title"]
            else:
                chunk["metadata"]["prev_section"] = None

            # 次のセクション
            if i < len(chunks) - 1:
                chunk["metadata"]["next_section"] = chunks[i+1]["metadata"]["title"]
            else:
                chunk["metadata"]["next_section"] = None

            # 相対位置
            chunk["metadata"]["relative_position"] = round(i / max(len(chunks) - 1, 1), 2)

    def _detect_code_block(self, text: str) -> bool:
        """コードブロックの検出"""
        code_indicators = ["```", "def ", "class ", "import ", "function ", "const ", "var ", "let "]
        return any(indicator in text for indicator in code_indicators)

    def _detect_speakers(self, text: str) -> List[str]:
        """発言者の検出（議事録用）"""
        import re
        # 「名前:」「名前：」パターンの検出
        speakers = re.findall(r'^([^:：\n]+)[：:]', text, re.MULTILINE)
        # 「【名前】」パターンの検出
        speakers.extend(re.findall(r'【([^】]+)】', text))
        # 重複を除去
        return list(set(speakers))[:5]  # 最大5人まで

    def _detect_numbers(self, text: str) -> bool:
        """数値データの検出（提案書用）"""
        import re
        # 金額、パーセント、期間などの検出
        patterns = [
            r'\d+円',
            r'\d+万円',
            r'\d+億円',
            r'\d+%',
            r'\d+％',
            r'\d+年',
            r'\d+ヶ月',
            r'\d+日間',
        ]
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False


class OptimizedVectorDBBuilder:
    """最適化されたベクトルDB構築処理（Phase 1改善）"""

    def __init__(self, file_cache: FileCache = None):
        # Embeddings初期化
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY が設定されていません")

        self.embeddings = GeminiEmbeddings(
            api_key=api_key,
            model_name=EMBEDDING_MODEL,
            dimension=DIMENSION
        )

        # Vector Store初期化
        self.vector_store = S3VectorStore(
            vector_bucket_name=VECTOR_BUCKET_NAME,
            index_name=VECTOR_INDEX_NAME,
            dimension=DIMENSION,
            distance_metric="cosine",
            region_name=AWS_REGION,
            create_if_not_exists=True,
        )

        # バッチ用バッファ
        self.chunk_buffer = []
        self.document_buffer = []

        # FileCacheへの参照を保持
        self.file_cache = file_cache

        # 進捗管理用カウンタ
        self.total_chunks = 0  # 全体のチャンク数
        self.processed_chunks = 0  # 処理済みチャンク数
        self.saved_chunks = 0  # S3に保存済みチャンク数

        # ファイル処理情報の管理
        self.file_processing_info = {}  # file_id -> {file_info, chunks_info}

    def save_chunks_batch(self, chunks: List[Dict], file_info: Dict, project_name: str = None) -> int:
        """チャンクをバッファに追加（Phase 1: バッチ処理）"""

        if not chunks:
            return 0

        file_id = file_info.get("id", "unknown")
        file_name = file_info.get("name", "unknown")
        log_prefix = f"[{file_name}]"

        # ファイル処理情報を記録
        if file_id not in self.file_processing_info:
            # ドキュメントタイプを判定
            document_type = "unknown"
            element_types = set()
            if chunks and len(chunks) > 0:
                for chunk in chunks:
                    if "elements" in chunk:
                        for elem in chunk.get("elements", []):
                            element_types.add(elem.get("type", "unknown"))
                if element_types:
                    document_type = ",".join(sorted(element_types))

            self.file_processing_info[file_id] = {
                "file_info": file_info,
                "project_name": project_name,
                "document_type": document_type,
                "chunk_count": 0,
                "saved": False
            }

        # 全体のチャンク数を更新
        new_chunks_count = len(chunks)
        self.total_chunks += new_chunks_count
        self.file_processing_info[file_id]["chunk_count"] += new_chunks_count
        print(f"    {log_prefix} [処理] {new_chunks_count}個のチャンクをバッファに追加... (全体進捗: {self.total_chunks}個)")

        for i, chunk_info in enumerate(chunks):
            self.chunk_buffer.append({
                "chunk": chunk_info,
                "file_name": file_name,
                "file_info": file_info,
                "file_id": file_id,
                "project_name": project_name,
                "chunk_index": i
            })

        # バッファが一定サイズに達したらバッチ処理
        if len(self.chunk_buffer) >= EMBEDDING_BATCH_SIZE:
            return self.process_batch()

        return 0

    def process_batch(self) -> int:
        """バッファのチャンクをバッチ処理（Phase 1改善）"""

        if not self.chunk_buffer:
            return 0

        batch_size = min(len(self.chunk_buffer), EMBEDDING_BATCH_SIZE)
        batch = self.chunk_buffer[:batch_size]
        self.chunk_buffer = self.chunk_buffer[batch_size:]

        # 全体の進捗を表示
        start_idx = self.processed_chunks + 1
        end_idx = self.processed_chunks + batch_size
        print(f"    [バッチ処理] チャンク {start_idx}-{end_idx}/{self.total_chunks}個をベクトル化中...")

        # テキストを抽出
        texts = [item["chunk"]["text"] for item in batch]

        # バッチでベクトル化（並列処理で高速化）
        try:
            # 並列処理でベクトル化（10並列）
            print("    [高速化] 並列処理でベクトル化中（10並列）...")
            vectors = self.embeddings.embed_documents_parallel(texts, max_workers=10)

            # 進捗更新
            current_total = self.processed_chunks + batch_size
            print(f"    [完了] ベクトル化: {batch_size}/{batch_size} (全体: {current_total}/{self.total_chunks})")

        except AttributeError:
            # embed_documents_parallelメソッドがない場合はフォールバック
            print("    [WARN] 並列処理メソッドが未実装。通常処理にフォールバック")
            vectors = []
            for i, text in enumerate(texts, 1):
                if i % 10 == 0:
                    current_total = self.processed_chunks + i
                    print(f"    [進捗] ベクトル化: {i}/{batch_size} (全体: {current_total}/{self.total_chunks})")
                vector = self.embeddings.embed_text(text)
                vectors.append(vector)

        except Exception as e:
            print(f"    [ERROR] バッチベクトル化エラー: {e}")
            # フォールバック：個別処理
            vectors = []
            for text in texts:
                try:
                    vector = self.embeddings.embed_text(text)
                    vectors.append(vector)
                except:
                    vectors.append(None)

        # Document作成
        documents = []
        seen_keys = set()  # 重複キーチェック用

        for idx, (item, vector) in enumerate(zip(batch, vectors)):
            # Noneチェックとゼロベクトルチェック
            if vector is None:
                print(f"    [WARN] インデックス {idx} のベクトルがNoneです。スキップします")
                continue

            # ゼロベクトルのチェック
            import numpy as np
            if isinstance(vector, list) and len(vector) > 0:
                vector_array = np.array(vector)
                vector_norm = np.linalg.norm(vector_array)
                if vector_norm == 0 or np.isclose(vector_norm, 0):
                    print(f"    [WARN] インデックス {idx} はゼロベクトルです。スキップします")
                    continue

            if vector and len(vector) == self.embeddings.dimension:
                chunk_info = item["chunk"]
                file_name = item["file_name"]
                metadata = chunk_info["metadata"]
                file_id = item.get("file_id", "unknown")
                project_name = item.get("project_name", "unknown")

                # より一意性の高いキー生成（ファイルIDと実際のインデックスを使用）
                # バッチ内でのインデックスも含めて一意性を保証
                batch_idx = self.processed_chunks + idx
                key_string = (
                    f"{project_name}/{file_id}/{file_name}/{metadata['chunk_index']}/{batch_idx}"
                )
                doc_key = f"doc_{hashlib.md5(key_string.encode()).hexdigest()[:16]}_{batch_idx}"

                # 重複チェック
                if doc_key in seen_keys:
                    # 重複の場合は追加のハッシュを付けて一意にする
                    import time
                    unique_suffix = f"_{int(time.time() * 1000) % 10000}_{idx}"
                    doc_key = f"{doc_key}{unique_suffix}"
                    print(f"    [WARN] キー重複検出、修正: {doc_key}")

                seen_keys.add(doc_key)

                # metadataにfile_idを追加
                metadata["file_id"] = file_id

                # Document作成
                doc = Document(
                    key=doc_key,
                    text=chunk_info["text"],
                    metadata=metadata
                )
                doc.vector = vector
                documents.append(doc)

        # バッチでS3保存（Phase 1改善）
        if documents:
            # 処理済みチャンク数を更新
            self.processed_chunks += batch_size
            self.saved_chunks += len(documents)

            print(f"    [バッチ処理] {len(documents)}個のドキュメントをS3に保存中... (全体進捗: {self.saved_chunks}/{self.total_chunks})")
            # TODO: batch_add_documentsメソッドが実装されていない場合
            # 現状は既存のadd_documentsメソッドを使用（内部でバッチ処理）
            added = self.vector_store.add_documents(documents, batch_size=S3_BATCH_SIZE)

            # S3保存成功後、ファイルごとにキャッシュに記録
            if self.file_cache and documents:
                # documentsから処理済みファイルIDを収集（実際に保存されたもののみ）
                processed_file_ids = set()
                for doc in documents:
                    # documentsのメタデータからfile_idを取得
                    if doc.metadata and "file_id" in doc.metadata:
                        processed_file_ids.add(doc.metadata["file_id"])

                # file_idが取得できない場合はbatchから取得を試みる
                if not processed_file_ids:
                    for item in batch:
                        if "file_id" in item:
                            processed_file_ids.add(item["file_id"])

                # 各ファイルについてmark_processedを呼び出し
                for file_id in processed_file_ids:
                    if file_id in self.file_processing_info:
                        file_data = self.file_processing_info[file_id]
                        if not file_data.get("saved", False):
                            # documentsからこのfile_idのdocument_typeを収集
                            document_type = 'unknown'
                            for doc in documents:
                                if doc.metadata and doc.metadata.get("file_id") == file_id:
                                    # document_typeから取得
                                    if "document_type" in doc.metadata:
                                        document_type = doc.metadata["document_type"]

                            # まだマークされていないファイルを処理済みとしてマーク
                            file_info = file_data["file_info"]
                            self.file_cache.mark_processed(
                                file_id=file_id,
                                modified_time=file_info.get("modifiedTime", ""),
                                chunk_count=file_data["chunk_count"],
                                project_name=file_data["project_name"],
                                file_name=file_info.get("name", ""),
                                document_type=document_type
                            )
                            file_data["saved"] = True
                            print(f"    [キャッシュ] ファイル {file_info.get('name', 'unknown')} を処理済みとして記録 (タイプ: {document_type})")

            # 完了率を表示
            if self.total_chunks > 0:
                completion_rate = (self.saved_chunks / self.total_chunks) * 100
                print(f"    [進捗] 全体の {completion_rate:.1f}% 完了 ({self.saved_chunks}/{self.total_chunks} チャンク)")

            return added

        # ドキュメントが作成されなかった場合も処理済みとしてカウント
        self.processed_chunks += batch_size
        return 0

    def flush_buffers(self) -> int:
        """残りのバッファをすべて処理"""
        total_added = 0

        if self.chunk_buffer:
            remaining = len(self.chunk_buffer)
            print(f"\n    [フラッシュ] 残り {remaining} 個のチャンクを処理中...")

        while self.chunk_buffer:
            added = self.process_batch()
            total_added += added

        # 最終統計を表示
        if self.total_chunks > 0:
            print("\n    [完了] 全チャンク処理完了:")
            print(f"      - 総チャンク数: {self.total_chunks}")
            print(f"      - 処理済み: {self.processed_chunks}")
            print(f"      - S3保存済み: {self.saved_chunks}")
            if self.saved_chunks < self.total_chunks:
                skipped = self.total_chunks - self.saved_chunks
                print(f"      - スキップ: {skipped} (エラーまたは無効なデータ)")

        return total_added

    def reset_counters(self):
        """進捗カウンターをリセット（新しいプロジェクト開始時用）"""
        self.total_chunks = 0
        self.processed_chunks = 0
        self.saved_chunks = 0


def _process_single_file_optimized(
    creds,
    chunker: OptimizedChunker,
    vector_db: OptimizedVectorDBBuilder,
    file_info: Dict,
    project_name: str,
) -> Optional[int]:
    """
    単一ファイルのRAG処理を行うワーカー関数（最適化版）

    Returns:
        保存されたドキュメント数、または None（失敗時）
    """
    import threading

    thread_id = threading.current_thread().name
    file_name = file_info['name']
    log_prefix = f"[Thread-{thread_id}][{file_name}]"

    try:
        print(f"{log_prefix} ファイル処理開始")

        # 各スレッド内で独自のserviceオブジェクトを生成
        service = build("drive", "v3", credentials=creds)

        # unstructuredでチャンク化（最適化版）
        chunks = chunker.chunk_file_with_unstructured(service, file_info, project_name)

        if chunks:
            # バッファに追加（バッチ処理）
            added = vector_db.save_chunks_batch(chunks, file_info, project_name)
            print(f"    {log_prefix} ✓ チャンクをバッファに追加")
            print(f"  {log_prefix} ======================================== \n")
            return len(chunks)  # チャンク数を返す

        print(f"  {log_prefix} ======================================== \n")
        return 0

    except Exception as e:
        print(f"{log_prefix} [ERROR] 処理中にエラーが発生: {e}")
        print(f"  {log_prefix} ======================================== \n")
        print(f"{log_prefix} エラー詳細:\n{traceback.format_exc()}")
        return None
    finally:
        print(f"{log_prefix} ファイル処理終了")


def main():
    """メイン処理"""
    print("=" * 60)
    print("LISA PoC - 最適化版 RAG ベクトルDB構築（Phase 1）")
    print("=" * 60)
    print()
    print("改善内容:")
    print("  - PDF戦略の事前判定（hi_res廃止）")
    print("  - バッチ処理（埋め込み・S3保存）")
    print("  - キャッシュ機構（処理済みスキップ）")
    print("  - 一時ファイル活用（メモリ削減）")
    print("  - マルチデータソース対応（Google Drive複数フォルダ）")
    print()

    # コマンドライン引数
    parser = argparse.ArgumentParser(description="最適化版 RAG ベクトルDB構築")
    parser.add_argument("--project", type=str, help="特定のプロジェクトのみ処理")
    parser.add_argument("--clear-cache", action="store_true", help="キャッシュをクリア")
    parser.add_argument("--config", type=str, default="project_config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()

    # キャッシュ初期化
    file_cache = FileCache()

    if args.clear_cache:
        file_cache.clear_cache()
        return

    # 環境変数チェック
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY が設定されていません")
        sys.exit(1)

    # プロジェクト設定を読み込み
    project_config = ProjectConfig(args.config)

    # 認証
    print("[INFO] OAuth認証中...")
    creds = authenticate()
    service = get_drive_service(creds)
    print("[INFO] 認証完了")

    # 処理コンポーネント初期化
    print("[INFO] 処理コンポーネント初期化中...")
    vector_db = OptimizedVectorDBBuilder(file_cache=file_cache)

    # ドキュメント分類器初期化
    print("[INFO] ドキュメント分類器を初期化中...")
    classifier = DocumentClassifier(
        api_key=api_key,
        project_config=project_config
    )
    print(f"[INFO] 分類器初期化完了（カテゴリ数: {len(classifier.categories)}）")

    chunker = OptimizedChunker(vector_db.embeddings, file_cache, classifier)
    print("[INFO] 初期化完了")

    # プロジェクト一覧を取得
    target_projects = []

    # 設定ファイルモード
    if project_config.is_config_loaded():
        print(f"[INFO] 設定ファイル読込完了: {project_config}")

        # プロジェクト一覧を取得
        if args.project:
            if project_config.has_project(args.project):
                project_names = [args.project]
            else:
                print(f"[ERROR] プロジェクト '{args.project}' が設定ファイルに見つかりません")
                sys.exit(1)
        else:
            project_names = project_config.get_projects()

        # 各プロジェクトのGoogle Driveフォルダを取得
        for project_name in project_names:
            folders = project_config.get_google_drive_folders(project_name)
            if folders:
                target_projects.append({
                    "name": project_name,
                    "folders": folders
                })
            else:
                print(f"[WARN] プロジェクト '{project_name}' にGoogle Driveフォルダが設定されていません")
    else:
        print("[ERROR] 設定ファイルが見つかりません。project_config.yamlを作成してください。")
        print("詳細は project_config.yaml.sample を参照してください。")
        sys.exit(1)

    if not target_projects:
        print("[WARN] 処理対象のプロジェクトが見つかりませんでした")
        sys.exit(0)

    print(
        f"[INFO] 処理対象: {', '.join([p['name'] for p in target_projects])} ({len(target_projects)}件)"
    )
    print()

    # 各プロジェクトを処理
    total_docs = 0
    total_chunks = 0

    for project in target_projects:
        project_name = project["name"]
        project_folders = project["folders"]

        print(f"\n{'=' * 50}")
        print(f"プロジェクト: {project_name}")
        print(f"Google Driveフォルダ数: {len(project_folders)}")
        print(f"{'=' * 50}")

        # プロジェクトごとに進捗カウンターをリセット
        vector_db.reset_counters()

        # 全フォルダからファイル一覧を取得
        all_files = []
        for folder_id in project_folders:
            print(f"[INFO] フォルダ {folder_id} をスキャン中...")
            files = list_files_in_folder(service, folder_id)
            # メタデータにdata_sourceを追加
            for file_info in files:
                file_info["data_source"] = "google_drive"
                file_info["folder_id"] = folder_id
            all_files.extend(files)

        print(f"[INFO] 合計 {len(all_files)}個のファイルを発見")

        project_chunks = 0

        # 並列処理でファイルを処理
        print(f"\n[INFO] 並列処理開始: {len(all_files)}ファイル, {MAX_WORKERS}ワーカー")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(
                    _process_single_file_optimized,
                    creds,
                    chunker,
                    vector_db,
                    file_info,
                    project_name,
                ): file_info
                for file_info in all_files
            }

            print(f"\n[INFO] {len(future_to_file)}個のタスクを投入しました")

            progress_bar = tqdm(
                as_completed(future_to_file),
                total=len(all_files),
                desc=f"Processing {project_name}",
            )
            completed_count = 0
            for future in progress_bar:
                file_info = future_to_file[future]
                completed_count += 1
                print(
                    f"\n[INFO] タスク完了 ({completed_count}/{len(all_files)}): {file_info['name']}"
                )
                try:
                    result = future.result()
                    if result is not None and result > 0:
                        project_chunks += result
                except Exception as exc:
                    print(f"[ERROR] '{file_info['name']}' の処理中に例外が発生: {exc}")
                    print(f"[ERROR] 例外詳細:\n{traceback.format_exc()}")

            print(f"\n[INFO] 全タスク完了: {len(files)}ファイル処理完了")

        # 残りのバッファを処理
        print("\n[INFO] 残りのバッファを処理中...")
        flushed = vector_db.flush_buffers()
        total_docs += flushed

        # メモリ解放
        gc.collect()

        total_chunks += project_chunks
        print(f"\n[INFO] {project_name}: {project_chunks}個のチャンクを処理")

    print()
    print("=" * 60)
    print("処理完了")
    print("=" * 60)
    print(f"総チャンク数: {total_chunks}個")
    print(f"総ドキュメント数（S3保存）: {total_docs}個")
    print()


if __name__ == "__main__":
    main()
