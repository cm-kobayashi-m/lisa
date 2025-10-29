"""
Gemini Embeddings API を使用したベクトル化処理

Google の Gemini API を使用してテキストをベクトル表現に変換します。
"""
import os
import json
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception
)
from httpx import ConnectError, TimeoutException

# ロギング設定
logger = logging.getLogger(__name__)


class GeminiQuotaError(Exception):
    """Gemini APIのクォータ制限エラー"""
    pass


class GeminiNetworkError(Exception):
    """Gemini APIのネットワーク接続エラー"""
    pass


class GeminiEmbeddings:
    """
    Gemini Embeddings API を使用したテキスト埋め込み

    特徴:
    - 1536次元のベクトル生成（デフォルト、768まで任意指定可能）
    - 多言語対応
    - 高品質な意味的表現
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/text-embedding-004",
        dimension: int = 1536,
        batch_size: int = 100
    ):
        """
        Gemini Embeddings を初期化

        Args:
            api_key: Gemini API キー（環境変数から取得可能）
            model_name: 使用する埋め込みモデル
            dimension: ベクトルの次元数（1536推奨、768まで任意指定可能）
            batch_size: バッチ処理のサイズ
        """
        # APIキーの取得
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY が設定されていません")

        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size

        # Geminiクライアントの初期化
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini Embeddings 初期化完了: {model_name}")
        except Exception as e:
            logger.error(f"Geminiクライアントの初期化に失敗: {e}")
            raise

    def _is_quota_error(self, exception: Exception) -> bool:
        """クォータエラーかどうかを判定"""
        error_msg = str(exception).lower()
        return any(keyword in error_msg for keyword in ['429', 'quota', 'rate limit'])

    def _is_network_error(self, exception: Exception) -> bool:
        """ネットワークエラーかどうかを判定"""
        # ConnectError, TimeoutException, DNS エラーなどを検出
        if isinstance(exception, (ConnectError, TimeoutException)):
            return True
        error_msg = str(exception).lower()
        network_keywords = [
            'nodename nor servname provided',
            'connection refused',
            'connection reset',
            'connection timeout',
            'name resolution',
            'dns',
            'network unreachable',
            'host unreachable'
        ]
        return any(keyword in error_msg for keyword in network_keywords)

    def _should_retry(self, exception: Exception) -> bool:
        """リトライすべきエラーかどうかを判定"""
        return self._is_quota_error(exception) or self._is_network_error(exception)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type((GeminiQuotaError, GeminiNetworkError)),
        before_sleep=lambda retry_state: logger.warning(
            f"API エラー検出 (試行 {retry_state.attempt_number}/5) - "
            f"{retry_state.next_action.sleep}秒待機してリトライします..."
        )
    )
    def embed_text(self, text: str) -> List[float]:
        """
        単一のテキストをベクトル化

        Args:
            text: ベクトル化するテキスト

        Returns:
            ベクトル表現（float配列）
        """
        if not text or not text.strip():
            logger.warning("空のテキストが渡されました")
            return [0.0] * self.dimension

        try:
            # 新しいGemini APIのフォーマット
            # configは辞書ではなくtypes.EmbedContentConfigオブジェクトを使用
            config = types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",  # RAG用の設定
                output_dimensionality=self.dimension
            )

            # Gemini Embeddings APIを呼び出し（引数名を修正）
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],  # contentではなくcontentsを使用し、リストで渡す
                config=config
            )

            # ベクトルを取得
            if response and hasattr(response, 'embeddings') and response.embeddings:
                # embeddingsは配列なので最初の要素を取得
                embedding = response.embeddings[0].values if response.embeddings[0].values else None

                if not embedding:
                    logger.error("APIレスポンスにベクトルデータが含まれていません")
                    return [0.0] * self.dimension

                # 次元数の確認
                if len(embedding) != self.dimension:
                    logger.warning(
                        f"期待される次元数 {self.dimension} と異なる次元数 {len(embedding)} "
                        f"のベクトルが返されました"
                    )

                return list(embedding)  # リストに変換して返す
            else:
                logger.error("APIレスポンスにembeddingsが含まれていません")
                return [0.0] * self.dimension

        except Exception as e:
            # エラーの種類を判定してリトライ可能な例外として再スロー
            if self._is_quota_error(e):
                logger.warning(f"クォータ制限エラー検出: {e}")
                raise GeminiQuotaError(str(e))
            elif self._is_network_error(e):
                logger.warning(f"ネットワークエラー検出: {e}")
                print(f"[DEBUG] ネットワークエラーの詳細: {type(e).__name__}: {e}")
                raise GeminiNetworkError(str(e))
            else:
                logger.error(f"テキストのベクトル化に失敗（リトライ不可）: {e}")
                # エラーの詳細をログに記録
                print(f"[DEBUG] エラーの詳細: {type(e).__name__}: {e}")
                raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        複数のテキストをバッチでベクトル化

        Args:
            texts: ベクトル化するテキストのリスト

        Returns:
            ベクトル表現のリスト
        """
        if not texts:
            return []

        embeddings = []
        total = len(texts)

        # バッチ処理
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []

            logger.info(
                f"バッチ処理中: {i+1}-{min(i+len(batch), total)}/{total}"
            )

            for text in batch:
                try:
                    # 個別にベクトル化（バッチAPIが利用可能になったら最適化）
                    embedding = self.embed_text(text)
                    batch_embeddings.append(embedding)

                    # レート制限対策として少し待機
                    time.sleep(0.1)

                except (GeminiQuotaError, GeminiNetworkError) as e:
                    # リトライ可能なエラーは embed_text() で既にリトライ済み
                    # ここに到達するのは全リトライ失敗後
                    logger.error(f"テキストのベクトル化に失敗（リトライ失敗）: {e}")
                    print(f"[WARN] ベクトル化エラー: {e}")
                    # エラー時はゼロベクトルで埋める
                    batch_embeddings.append([0.0] * self.dimension)
                except Exception as e:
                    logger.error(f"テキストのベクトル化に失敗（予期しないエラー）: {e}")
                    print(f"[ERROR] 予期しないベクトル化エラー: {type(e).__name__}: {e}")
                    # エラー時はゼロベクトルで埋める
                    batch_embeddings.append([0.0] * self.dimension)

            embeddings.extend(batch_embeddings)

        logger.info(f"{len(embeddings)}件のテキストをベクトル化しました")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        クエリテキストをベクトル化（検索用）

        Args:
            text: クエリテキスト

        Returns:
            ベクトル表現
        """
        if not text or not text.strip():
            logger.warning("空のクエリテキストが渡されました")
            return [0.0] * self.dimension

        try:
            # 新しいGemini APIのフォーマット
            config = types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",  # クエリ用の設定
                output_dimensionality=self.dimension
            )

            # クエリ用の設定でベクトル化
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],  # contentではなくcontentsを使用し、リストで渡す
                config=config
            )

            if response and hasattr(response, 'embeddings') and response.embeddings:
                # embeddingsは配列なので最初の要素を取得
                embedding = response.embeddings[0].values if response.embeddings[0].values else None

                if not embedding:
                    logger.error("APIレスポンスにベクトルデータが含まれていません")
                    return [0.0] * self.dimension

                return list(embedding)  # リストに変換して返す
            else:
                logger.error("APIレスポンスにembeddingsが含まれていません")
                return [0.0] * self.dimension

        except Exception as e:
            # エラーの種類を判定してリトライ可能な例外として再スロー
            if self._is_quota_error(e):
                logger.warning(f"クォータ制限エラー検出: {e}")
                raise GeminiQuotaError(str(e))
            elif self._is_network_error(e):
                logger.warning(f"ネットワークエラー検出: {e}")
                print(f"[DEBUG] ネットワークエラーの詳細: {type(e).__name__}: {e}")
                raise GeminiNetworkError(str(e))
            else:
                logger.error(f"クエリのベクトル化に失敗（リトライ不可）: {e}")
                print(f"[DEBUG] エラーの詳細: {type(e).__name__}: {e}")
                raise

    def cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        2つのベクトル間のコサイン類似度を計算

        Args:
            vec1: ベクトル1
            vec2: ベクトル2

        Returns:
            コサイン類似度（-1.0 〜 1.0）
        """
        if len(vec1) != len(vec2):
            raise ValueError(
                f"ベクトルの次元が一致しません: {len(vec1)} != {len(vec2)}"
            )

        # NumPyを使用してコサイン類似度を計算
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        # ゼロベクトルのチェック
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # コサイン類似度
        return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))

    def chunk_text(
        self,
        text: str,
        max_chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """
        長いテキストを重複ありでチャンクに分割

        Args:
            text: 分割するテキスト
            max_chunk_size: 最大チャンクサイズ（文字数）
            overlap: チャンク間の重複サイズ

        Returns:
            チャンクのリスト
        """
        if not text:
            return []

        chunks = []
        text_length = len(text)

        # オーバーラップを考慮してチャンクを作成
        start = 0
        while start < text_length:
            end = min(start + max_chunk_size, text_length)

            # 文の境界で分割を試みる
            if end < text_length:
                # 文末を探す（。、！、？）
                for delimiter in ['。', '！', '？', '\n\n', '\n', ' ']:
                    last_delimiter = text.rfind(delimiter, start, end)
                    if last_delimiter != -1:
                        end = last_delimiter + len(delimiter)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 次の開始位置（オーバーラップを考慮）
            start = end - overlap if end < text_length else text_length

        logger.info(f"テキストを{len(chunks)}個のチャンクに分割しました")
        return chunks

    def create_document_embeddings(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        ドキュメントをチャンク化してベクトル化

        Args:
            text: ドキュメントテキスト
            metadata: ドキュメントのメタデータ
            chunk_size: チャンクサイズ
            overlap: オーバーラップサイズ

        Returns:
            チャンクとベクトルを含む辞書のリスト
        """
        # チャンク化
        chunks = self.chunk_text(text, chunk_size, overlap)

        # ベクトル化
        embeddings = self.embed_documents(chunks)

        # 結果を構築
        results = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)

            results.append({
                'text': chunk,
                'embedding': embedding,
                'metadata': chunk_metadata
            })

        return results