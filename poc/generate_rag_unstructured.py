#!/usr/bin/env python3
"""
LISA PoC - unstructuredライブラリベースのRAG用ベクトルDB構築スクリプト

unstructuredライブラリを使用して、PDFやOfficeファイルをローカルで処理し、
チャンク化とベクトル化を行い、S3 Vectorsに保存します。

使用方法:
    # すべてのプロジェクトを処理
    python3 generate_rag_unstructured.py

    # 特定のプロジェクトのみ処理
    python3 generate_rag_unstructured.py --project "LISAのPoCテスト"

特徴:
    - unstructuredライブラリによるローカル処理（LLM APIコスト削減）
    - 構造を活かしたインテリジェントなチャンク分割
    - MS Office形式の直接サポート
    - PDFの二段階処理戦略（fast → 必要時hi_res）
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
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Google Drive API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

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
from unstructured.partition.image import partition_image
from unstructured.chunking.title import chunk_by_title

# テキスト分割用
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RAGモジュール
from rag.vector_store import S3VectorStore, Document
from rag.embeddings import GeminiEmbeddings

# 定数
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "token.yaml"
CREDENTIALS_FILE = "credentials.json"
TEMP_DIR = "temp_files"
MAX_FILE_SIZE = 30  # MB（PDF/Officeファイル用に増加）
MAX_WORKERS = 20  # 並列処理のワーカー数（ローカル処理なので増やせる）

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
    target_mime_types = [
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


class UnstructuredChunker:
    """unstructuredライブラリによるチャンク化処理"""

    def __init__(self, embeddings: GeminiEmbeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def chunk_file_with_unstructured(
        self, service, file_info: Dict, project_name: str
    ) -> List[Dict]:
        """
        unstructuredライブラリでファイルを読み込み、チャンク化する

        Returns:
            チャンク情報のリスト（text, metadata）
        """
        file_id = file_info["id"]
        file_name = file_info["name"]
        mime_type = file_info["mimeType"]

        print(f"\n  ========================================")
        print(f"  処理中: {file_name}")
        print(f"  ========================================")

        # ファイルサイズチェック
        file_size = int(file_info.get("size", 0))
        print(
            f"    [情報] ファイルサイズ: {file_size / 1024:.1f}KB, MIMEタイプ: {mime_type}"
        )

        if file_size > MAX_FILE_SIZE * 1024 * 1024:
            print(
                f"    [SKIP] ファイルサイズが大きすぎます: {file_size / 1024 / 1024:.1f}MB"
            )
            return []

        try:
            # ファイルをダウンロード（適切な形式で）
            print(f"    [開始] ダウンロード処理を開始...")
            download_result = self._download_file(
                service, file_id, file_name, mime_type
            )

            if not download_result:
                print(f"    [SKIP] ファイルを読み込めませんでした")
                return []

            file_content, actual_mime_type = download_result

            # unstructuredでElements抽出
            elements = self._partition_elements(
                file_content, actual_mime_type, file_name
            )

            if not elements:
                print(f"    [SKIP] コンテンツを抽出できませんでした")
                return []

            # 構造的なチャンク化
            chunks = self._elements_to_chunks(elements, project_name, file_name)

            # ファイルのメタデータを追加
            for chunk in chunks:
                if "modifiedTime" in file_info:
                    chunk["metadata"]["modified_at"] = file_info["modifiedTime"]
                if "createdTime" in file_info:
                    chunk["metadata"]["created_at"] = file_info["createdTime"]

            print(f"    {len(chunks)}個のチャンクを生成")
            return chunks

        except Exception as e:
            print(f"    [ERROR] 処理エラー: {e}")
            print(f"    [DEBUG] エラー詳細: {traceback.format_exc()}")
            return []

    def _download_file(
        self, service, file_id: str, file_name: str, mime_type: str
    ) -> Optional[Tuple[bytes, str]]:
        """ファイルをダウンロード（適切な形式で）"""
        try:
            # Google Docs → DOCX
            if mime_type == "application/vnd.google-apps.document":
                print(f"    [ダウンロード] Google DocsをDOCX形式でエクスポート中...")
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(
                    f"    [ダウンロード] 完了 ({len(file_stream.getvalue()) / 1024 / 1024:.2f}MB)"
                )
                return (
                    file_stream.getvalue(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

            # Google Sheets → XLSX
            elif mime_type == "application/vnd.google-apps.spreadsheet":
                print(f"    [ダウンロード] Google SheetsをXLSX形式でエクスポート中...")
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(
                    f"    [ダウンロード] 完了 ({len(file_stream.getvalue()) / 1024 / 1024:.2f}MB)"
                )
                return (
                    file_stream.getvalue(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            # Google Slides → PPTX
            elif mime_type == "application/vnd.google-apps.presentation":
                print(f"    [ダウンロード] Google SlidesをPPTX形式でエクスポート中...")
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(
                    f"    [ダウンロード] 完了 ({len(file_stream.getvalue()) / 1024 / 1024:.2f}MB)"
                )
                return (
                    file_stream.getvalue(),
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )

            # その他のファイル（PDF、MS Office、画像など）はそのままダウンロード
            else:
                print(f"    [ダウンロード] ファイルダウンロード中...")
                request = service.files().get_media(fileId=file_id)
                file_stream = io.BytesIO()
                downloader = MediaIoBaseDownload(file_stream, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(
                    f"    [ダウンロード] 完了 ({len(file_stream.getvalue()) / 1024 / 1024:.2f}MB)"
                )
                return file_stream.getvalue(), mime_type

        except Exception as e:
            print(f"    [ERROR] ダウンロードエラー: {e}")
            return None

    def _process_image_with_gemini(
        self, file_bytes: bytes, mime_type: str, file_name: str
    ) -> List[Any]:
        """LangChain経由でGemini LLMを使用して画像からテキストを抽出（OCR）"""
        try:
            # API キーの確認
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print(
                    f"    [WARN] GEMINI_API_KEY が設定されていないため、画像処理をスキップ"
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
                # テキストをElement風のオブジェクトに変換
                # （unstructuredの他の処理と互換性を保つため）
                class SimpleElement:
                    def __init__(self, text, category="Text"):
                        self.text = text
                        self.category = category
                        self.metadata = type("obj", (object,), {"page_number": 1})()

                # 段落ごとに分割してElement化
                paragraphs = response.content.strip().split("\n\n")
                elements = []
                for para in paragraphs:
                    if para.strip():
                        elements.append(SimpleElement(para.strip()))

                print(f"    [処理] LangChain Gemini OCRで{len(elements)}個の要素を抽出")
                return elements
            else:
                print(f"    [WARN] LangChain Geminiからのレスポンスが空です")
                return []

        except Exception as e:
            print(f"    [ERROR] LangChain Gemini OCR処理エラー: {e}")
            print(f"    [DEBUG] {traceback.format_exc()}")
            return []

    def _partition_elements(
        self, file_bytes: bytes, mime_type: str, file_name: str
    ) -> List[Any]:
        """ファイル形式に応じたpartition処理"""
        try:
            f = io.BytesIO(file_bytes)

            # PDF
            if mime_type == "application/pdf":
                print(f"    [処理] PDFを解析中 (fast戦略)...")
                # 二段階戦略: fast → 必要時hi_res
                elements = partition_pdf(
                    file=f, strategy="fast", metadata_filename=file_name
                )

                # テキストが少ない場合の判定
                total_text_length = sum(
                    len(getattr(e, "text", "") or "") for e in elements
                )
                if total_text_length < 300:
                    print(f"    [処理] テキストが少ないため、hi_res戦略で再処理...")
                    f.seek(0)
                    elements = partition_pdf(
                        file=f,
                        strategy="hi_res",
                        infer_table_structure=True,
                        extract_images_in_pdf=False,
                        metadata_filename=file_name,
                    )
                return elements

            # MS Word / Google Docs (DOCX形式)
            elif mime_type in OFFICE_WORD_MIMES:
                print(f"    [処理] Word/Docsドキュメントを解析中...")
                return partition_docx(file=f, metadata_filename=file_name)

            # MS Excel / Google Sheets (XLSX形式)
            elif mime_type in OFFICE_EXCEL_MIMES:
                print(f"    [処理] Excel/Sheetsスプレッドシートを解析中...")
                return partition_xlsx(file=f, metadata_filename=file_name)

            # MS PowerPoint / Google Slides (PPTX形式)
            elif mime_type in OFFICE_PPT_MIMES:
                print(f"    [処理] PowerPoint/Slidesプレゼンテーションを解析中...")
                return partition_pptx(file=f, metadata_filename=file_name)

            # Markdown
            elif mime_type in ["text/markdown", "text/x-markdown"]:
                print(f"    [処理] Markdownドキュメントを解析中...")
                return partition_md(file=f, metadata_filename=file_name)

            # プレーンテキスト
            elif mime_type.startswith("text/"):
                print(f"    [処理] テキストドキュメントを解析中...")
                return partition_text(file=f, metadata_filename=file_name)

            # 画像（Gemini LLMでOCR処理）
            elif mime_type in ["image/png", "image/jpeg", "image/jpg"]:
                print(f"    [処理] 画像ファイルをGemini LLMでOCR処理中...")
                return self._process_image_with_gemini(file_bytes, mime_type, file_name)

            else:
                print(f"    [WARN] 未対応のMIMEタイプ: {mime_type}")
                return []

        except Exception as e:
            print(f"    [ERROR] Elements抽出エラー: {e}")
            print(f"    [DEBUG] {traceback.format_exc()}")
            return []

    def _elements_to_chunks(
        self, elements: List[Any], project_name: str, file_name: str
    ) -> List[Dict]:
        """Elementsを構造的にチャンク化"""
        try:
            print(f"    [処理] {len(elements)}個のElementsをチャンク化中...")

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
                    # カテゴリを収集
                    categories = set()
                    pages = set()
                    if hasattr(tc, "elements"):
                        for elem in tc.elements:
                            if hasattr(elem, "category") and elem.category:
                                categories.add(elem.category)
                            if hasattr(elem, "metadata") and hasattr(
                                elem.metadata, "page_number"
                            ):
                                pages.add(elem.metadata.page_number)

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
                                "categories": list(categories) if categories else [],
                                "pages": sorted(list(pages)) if pages else [],
                                "importance": "medium",
                            },
                        }
                    )

            return chunks

        except Exception as e:
            print(f"    [ERROR] チャンク化エラー: {e}")
            print(f"    [DEBUG] {traceback.format_exc()}")
            # フォールバック：単純な分割
            return self._simple_chunk(elements, project_name, file_name)

    def _simple_chunk(
        self, elements: List[Any], project_name: str, file_name: str
    ) -> List[Dict]:
        """フォールバック用の単純なチャンク分割"""
        try:
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
                            "categories": [],
                            "pages": [],
                            "importance": "medium",
                        },
                    }
                )

            return chunks
        except Exception as e:
            print(f"    [ERROR] フォールバックチャンク化エラー: {e}")
            return []


class VectorDBBuilder:
    """ベクトルDB構築処理"""

    def __init__(self):
        # Embeddings初期化
        api_key = os.getenv("GEMINI_API_KEY")
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
            create_if_not_exists=True,
        )

    def save_chunks_to_vector_db(self, chunks: List[Dict], file_name: str) -> int:
        """チャンクをベクトル化してS3 Vectorsに保存"""

        if not chunks:
            return 0

        print(f"    [処理] {len(chunks)}個のチャンクをベクトル化中...")
        documents = []

        for i, chunk_info in enumerate(chunks, 1):
            # 進捗表示（10個ごと）
            if i % 10 == 0 or i == len(chunks):
                print(f"    [進捗] ベクトル化: {i}/{len(chunks)}")

            text = chunk_info["text"]
            metadata = chunk_info["metadata"]

            # ドキュメントキー生成
            key_string = (
                f"{metadata['project_name']}/{file_name}/{metadata['chunk_index']}"
            )
            doc_key = f"doc_{hashlib.md5(key_string.encode()).hexdigest()[:16]}_{metadata['chunk_index']}"

            # Document作成
            doc = Document(key=doc_key, text=text, metadata=metadata)

            # ベクトル化
            try:
                doc.vector = self.embeddings.embed_text(text)
                if doc.vector and len(doc.vector) == self.embeddings.dimension:
                    documents.append(doc)
            except Exception as e:
                print(f"    [WARN] ベクトル化エラー: {e}")

        # ベクトルストアに保存
        if documents:
            print(f"    [処理] {len(documents)}個のドキュメントをS3 Vectorsに保存中...")
            added = self.vector_store.add_documents(documents, batch_size=5)
            return added

        return 0


def _process_single_file_rag(
    creds,
    chunker: UnstructuredChunker,
    vector_db: VectorDBBuilder,
    file_info: Dict,
    project_name: str,
) -> Optional[int]:
    """
    単一ファイルのRAG処理を行うワーカー関数。

    Returns:
        保存されたドキュメント数、または None（失敗時）
    """
    import threading

    thread_id = threading.current_thread().name

    try:
        print(f"[Thread-{thread_id}] ファイル処理開始: {file_info['name']}")

        # 各スレッド内で独自のserviceオブジェクトを生成
        service = build("drive", "v3", credentials=creds)

        # unstructuredでチャンク化
        chunks = chunker.chunk_file_with_unstructured(service, file_info, project_name)

        if chunks:
            # ベクトルDBに保存
            added = vector_db.save_chunks_to_vector_db(chunks, file_info["name"])
            print(f"    ✓ {added}個のドキュメントを保存完了")
            print(f"  ======================================== \n")
            return added

        print(f"  ======================================== \n")
        return 0

    except Exception as e:
        print(
            f"[Thread-{thread_id}] [ERROR] '{file_info['name']}' の処理中にエラーが発生: {e}"
        )
        print(f"  ======================================== \n")
        print(f"[Thread-{thread_id}] エラー詳細:\n{traceback.format_exc()}")
        return None
    finally:
        print(f"[Thread-{thread_id}] ファイル処理終了: {file_info['name']}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("LISA PoC - unstructured RAG ベクトルDB構築")
    print("=" * 60)
    print()
    print("特徴:")
    print("  - unstructuredライブラリによるローカル処理")
    print("  - 構造を活かしたインテリジェントなチャンク分割")
    print("  - MS Office形式の直接サポート")
    print("  - コスト効率的な処理（LLM API不要）")
    print()

    # コマンドライン引数
    parser = argparse.ArgumentParser(description="unstructured RAG ベクトルDB構築")
    parser.add_argument("--project", type=str, help="特定のプロジェクトのみ処理")
    args = parser.parse_args()

    # 環境変数チェック
    api_key = os.getenv("GEMINI_API_KEY")
    projects_folder_id = os.getenv("PROJECTS_FOLDER_ID")
    project_names = os.getenv("PROJECT_NAMES", "*")

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

    # 処理コンポーネント初期化
    print("[INFO] 処理コンポーネント初期化中...")
    vector_db = VectorDBBuilder()
    chunker = UnstructuredChunker(vector_db.embeddings)
    print("[INFO] 初期化完了")

    # プロジェクト一覧取得
    all_projects = list_project_folders(service, projects_folder_id)

    # フィルタリング
    if args.project:
        target_projects = [p for p in all_projects if p["name"] == args.project]
    else:
        if project_names != "*":
            target_names = [name.strip() for name in project_names.split(",")]
            target_projects = [p for p in all_projects if p["name"] in target_names]
        else:
            target_projects = all_projects

    if not target_projects:
        print("[WARN] 処理対象のプロジェクトが見つかりませんでした")
        sys.exit(0)

    print(
        f"[INFO] 処理対象: {', '.join([p['name'] for p in target_projects])} ({len(target_projects)}件)"
    )
    print()

    # 各プロジェクトを処理
    total_docs = 0

    for project in target_projects:
        project_name = project["name"]
        project_id = project["id"]

        print(f"\n{'=' * 50}")
        print(f"プロジェクト: {project_name}")
        print(f"{'=' * 50}")

        # ファイル一覧取得
        files = list_files_in_folder(service, project_id)
        print(f"[INFO] {len(files)}個のファイルを発見")

        project_docs = 0

        # 並列処理でファイルを処理
        print(f"\n[INFO] 並列処理開始: {len(files)}ファイル, {MAX_WORKERS}ワーカー")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(
                    _process_single_file_rag,
                    creds,
                    chunker,
                    vector_db,
                    file_info,
                    project_name,
                ): file_info
                for file_info in files
            }

            print(f"[INFO] {len(future_to_file)}個のタスクを投入しました")

            progress_bar = tqdm(
                as_completed(future_to_file),
                total=len(files),
                desc=f"Processing {project_name}",
            )
            completed_count = 0
            for future in progress_bar:
                file_info = future_to_file[future]
                completed_count += 1
                print(
                    f"\n[INFO] タスク完了 ({completed_count}/{len(files)}): {file_info['name']}"
                )
                try:
                    result = future.result()
                    if result is not None and result > 0:
                        project_docs += result
                except Exception as exc:
                    print(f"[ERROR] '{file_info['name']}' の処理中に例外が発生: {exc}")
                    print(f"[ERROR] 例外詳細:\n{traceback.format_exc()}")

            print(f"\n[INFO] 全タスク完了: {len(files)}ファイル処理完了")

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
