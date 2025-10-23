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
import io
import yaml
import json
import argparse
import gc
import traceback
import hashlib
import tempfile
import pickle
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
from pathlib import Path

# Google Drive API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# PyMuPDF for PDF text detection
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("[WARN] PyMuPDF not installed. PDF text detection disabled.")

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
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """キャッシュファイルを保存"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get_file_hash(self, file_id: str, modified_time: str) -> str:
        """ファイルのハッシュ値を生成"""
        hash_input = f"{file_id}:{modified_time}:{CACHE_VERSION}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def is_processed(self, file_id: str, modified_time: str) -> bool:
        """ファイルが処理済みか確認"""
        file_hash = self.get_file_hash(file_id, modified_time)
        return file_hash in self.cache

    def mark_processed(self, file_id: str, modified_time: str, chunk_count: int):
        """ファイルを処理済みとしてマーク"""
        file_hash = self.get_file_hash(file_id, modified_time)
        self.cache[file_hash] = {
            "file_id": file_id,
            "modified_time": modified_time,
            "processed_at": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "cache_version": CACHE_VERSION
        }
        self._save_cache()

    def clear_cache(self):
        """キャッシュをクリア"""
        self.cache = {}
        self._save_cache()
        print(f"[INFO] キャッシュをクリアしました")


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
        .list(q=query, fields="files(id, name)", orderBy="name")
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

    def check_pdf_has_text(self, file_path: str) -> bool:
        """PDFにテキストが含まれているか事前チェック（Phase 1改善）"""
        if not PYMUPDF_AVAILABLE:
            return False  # PyMuPDFがなければfalse（hi_res回避）

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

        print(f"\n  ========================================")
        print(f"  {log_prefix} 処理中")
        print(f"  ========================================")

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

            # キャッシュに記録
            self.file_cache.mark_processed(file_id, modified_time, len(chunks))

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
                    request = service.files().get_media(fileId=file_id)

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
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1,  # 低めの温度で正確性重視
                max_output_tokens=4096,
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
                paragraphs = response.content.strip().split("\n\n")
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
        """Elementsを構造的にチャンク化"""
        log_prefix = f"[{file_name}]"
        try:
            print(f"    {log_prefix} [処理] {len(elements)}個のElementsをチャンク化中...")

            # ドキュメント分類（1ファイルにつき1回のLLM呼び出し）
            print(f"    {log_prefix} [分類] ドキュメント種別を判定中...")
            document_type, confidence = self.classifier.classify_from_elements(
                elements=elements,
                file_name=file_name,
                max_elements=10  # 先頭10要素から判定
            )
            print(f"    {log_prefix} [分類] 判定結果: {document_type} (信頼度: {confidence:.2f})")

            # chunk_by_titleで論理単位にグループ化
            title_chunks = chunk_by_title(
                elements,
                max_characters=1200,
                new_after_n_chars=1000,
                combine_text_under_n_chars=200,
                multipage_sections=True,
            )

            chunks = []
            for idx, tc in enumerate(title_chunks):
                # テキストを結合
                text_parts = []
                for element in tc.elements if hasattr(tc, "elements") else [tc]:
                    elem_text = getattr(element, "text", "") or ""
                    if elem_text:
                        text_parts.append(elem_text)

                text = " ".join(text_parts).strip()
                if not text:
                    continue

                # 長すぎる場合は追加で分割
                if len(text) > 1400:
                    sub_texts = self.text_splitter.split_text(text)
                else:
                    sub_texts = [text]

                # チャンク作成
                for sub_idx, sub_text in enumerate(sub_texts):
                    # Element種別を取得（Title, Text等）
                    element_types = set()
                    if hasattr(tc, "category"):
                        element_types.add(tc.category)

                    # ページを取得
                    pages = set()
                    if hasattr(tc, "metadata") and hasattr(tc.metadata, "page_number"):
                        pages.add(tc.metadata.page_number)

                    # タイトルを取得
                    title = ""
                    if hasattr(tc, "title"):
                        title = tc.title
                    elif hasattr(tc, "metadata") and hasattr(tc.metadata, "title"):
                        title = tc.metadata.title

                    chunks.append(
                        {
                            "text": sub_text,
                            "metadata": {
                                "project_name": project_name,
                                "file_name": file_name,
                                "chunk_index": len(chunks),
                                "title": title.strip()
                                if title
                                else f"Section {idx + 1}",
                                # ドキュメント種別（LLM判定）
                                "document_type": document_type,
                                "document_type_confidence": confidence,
                                # Element種別（unstructured判定）
                                "element_types": list(element_types) if element_types else [],
                                "pages": sorted(list(pages)) if pages else [],
                                "importance": "medium",
                            },
                        }
                    )

            return chunks

        except Exception as e:
            log_prefix = f"[{file_name}]"
            print(f"    {log_prefix} [ERROR] チャンク化エラー: {e}")
            print(f"    {log_prefix} [DEBUG] {traceback.format_exc()}")
            # フォールバック：単純な分割
            return self._simple_chunk(elements, project_name, file_name)

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


class OptimizedVectorDBBuilder:
    """最適化されたベクトルDB構築処理（Phase 1改善）"""

    def __init__(self):
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

    def save_chunks_batch(self, chunks: List[Dict], file_name: str) -> int:
        """チャンクをバッファに追加（Phase 1: バッチ処理）"""

        if not chunks:
            return 0

        log_prefix = f"[{file_name}]"
        print(f"    {log_prefix} [処理] {len(chunks)}個のチャンクをバッファに追加...")

        for chunk_info in chunks:
            self.chunk_buffer.append({
                "chunk": chunk_info,
                "file_name": file_name
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

        print(f"    [バッチ処理] {batch_size}個のチャンクをベクトル化中...")

        # テキストを抽出
        texts = [item["chunk"]["text"] for item in batch]

        # バッチでベクトル化（Phase 1: API呼び出し削減）
        try:
            # TODO: embed_batchメソッドが実装されていない場合は個別処理
            # 現状は個別処理（将来的にバッチ対応予定）
            vectors = []
            for i, text in enumerate(texts, 1):
                if i % 10 == 0:
                    print(f"    [進捗] ベクトル化: {i}/{len(texts)}")
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
        for item, vector in zip(batch, vectors):
            if vector and len(vector) == self.embeddings.dimension:
                chunk_info = item["chunk"]
                file_name = item["file_name"]
                metadata = chunk_info["metadata"]

                # ドキュメントキー生成
                key_string = (
                    f"{metadata['project_name']}/{file_name}/{metadata['chunk_index']}"
                )
                doc_key = f"doc_{hashlib.md5(key_string.encode()).hexdigest()[:16]}_{metadata['chunk_index']}"

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
            print(f"    [バッチ処理] {len(documents)}個のドキュメントをS3に保存中...")
            # TODO: batch_add_documentsメソッドが実装されていない場合
            # 現状は既存のadd_documentsメソッドを使用（内部でバッチ処理）
            added = self.vector_store.add_documents(documents, batch_size=S3_BATCH_SIZE)
            return added

        return 0

    def flush_buffers(self) -> int:
        """残りのバッファをすべて処理"""
        total_added = 0

        while self.chunk_buffer:
            added = self.process_batch()
            total_added += added

        return total_added


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
            added = vector_db.save_chunks_batch(chunks, file_info["name"])
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
    vector_db = OptimizedVectorDBBuilder()

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
        print(f"\n[INFO] 残りのバッファを処理中...")
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
