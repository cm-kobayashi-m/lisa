"""
RAG検索を行うヘルパークラス

S3 Vectorsから類似ドキュメントを検索し、プロンプトに統合するための機能を提供します。
"""

import os
import logging
import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .vector_store import S3VectorStore, Document
from .embeddings import GeminiEmbeddings

# ロギング設定
logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG検索を行うヘルパークラス"""

    def __init__(self, vector_store: S3VectorStore, embeddings: GeminiEmbeddings):
        """
        RAGRetrieverを初期化

        Args:
            vector_store: S3VectorStoreインスタンス
            embeddings: GeminiEmbeddingsインスタンス
        """
        self.vector_store = vector_store
        self.embeddings = embeddings

    def search_similar_documents(
        self, query: str, project_name: Optional[str] = None, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        類似ドキュメントを検索

        Args:
            query: 検索クエリ
            project_name: プロジェクト名でフィルタ（オプション）
            file_name: ファイル名でフィルタ（オプション）
            k: 返す結果数（デフォルト: 5）

        Returns:
            (Document, 類似度スコア)のタプルのリスト
        """
        try:
            # クエリをベクトル化
            logger.info(f"クエリをベクトル化中: {query[:100]}...")
            query_vector = self.embeddings.embed_query(query)

            # フィルタ条件の構築
            filter_dict = None
            if project_name:
                filter_dict = {"project_name": {"$eq": project_name}}
                logger.info(f"プロジェクト名でフィルタ: {project_name}")

            # 類似度検索
            logger.info(f"S3 Vectorsで類似度検索を実行（k={k}）")
            results = self.vector_store.similarity_search(
                query_vector=query_vector, k=k, filter_dict=filter_dict
            )

            logger.info(f"検索完了: {len(results)}件の結果を取得")
            return results

        except Exception as e:
            logger.error(f"類似ドキュメント検索でエラーが発生: {e}")
            # エラーが発生した場合は空のリストを返す
            return []

    def format_context_for_prompt(
        self,
        results: List[Tuple[Document, float]],
        max_chars: int = 5000,
        include_metadata: bool = True,
    ) -> str:
        """
        検索結果をプロンプト用にフォーマット

        Args:
            results: 検索結果（Document, 類似度スコア）のリスト
            max_chars: 最大文字数（デフォルト: 5000）
            include_metadata: メタデータを含むか（デフォルト: True）

        Returns:
            フォーマット済みのコンテキスト文字列
        """
        if not results:
            return ""

        context = "## 関連情報（RAG検索結果）\n\n"
        context += (
            "以下は過去の類似プロジェクトや関連ドキュメントから抽出された情報です。\n"
        )
        context += "これらの情報を参考にして、より深い分析を行ってください。\n\n"

        total_chars = len(context)

        for i, (doc, score) in enumerate(results, 1):
            # S3 Vectorsの距離を類似度に変換
            # S3 Vectorsは類似度ではなく距離を返すため、メトリックに応じて正規化
            distance_metric = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")
            if distance_metric == "cosine":
                # cosine距離は 1 - cosine_similarity
                similarity = max(0.0, min(1.0, 1.0 - score))
            elif distance_metric == "euclidean":
                # euclidean距離を類似度に変換
                similarity = 1.0 / (1.0 + score)
            else:
                # フォールバック: 既存の計算方法（互換性のため）
                similarity = (1 + score) / 2

            similarity_percent = similarity * 100

            # ドキュメントのテキストをフォーマット
            doc_text = f"### {i}. 関連度: {similarity_percent:.1f}%\n"

            if include_metadata:
                # メタデータを含める
                doc_text += (
                    f"- **プロジェクト**: {doc.metadata.get('project_name', '不明')}\n"
                )
                doc_text += f"- **ファイル**: {doc.metadata.get('file_name', '不明')}\n"

                # その他のメタデータ（あれば）
                if "chunk_index" in doc.metadata:
                    doc_text += f"- **チャンク番号**: {doc.metadata['chunk_index']}\n"
                if "title" in doc.metadata:
                    doc_text += f"- **タイトル**: {doc.metadata['title']}\n"
                if "topics" in doc.metadata and doc.metadata["topics"]:
                    doc_text += f"- **トピック**: {', '.join(doc.metadata['topics'])}\n"
                if "importance" in doc.metadata:
                    doc_text += f"- **重要度**: {doc.metadata['importance']}\n"

            # ドキュメントの内容（文字数制限を考慮）
            remaining_chars = max_chars - total_chars - len(doc_text) - 100  # バッファ
            if remaining_chars > 100:
                content_limit = min(500, remaining_chars)
                truncated_text = doc.text[:content_limit]
                if len(doc.text) > content_limit:
                    truncated_text += "..."
                doc_text += f"\n**内容**:\n```\n{truncated_text}\n```\n\n"
            else:
                # 文字数制限に達した場合
                context += "\n（以降、文字数制限により省略）\n"
                break

            # 文字数チェック
            if total_chars + len(doc_text) > max_chars:
                context += f"\n（文字数制限により、残り{len(results) - i + 1}件の結果を省略）\n"
                break

            context += doc_text
            total_chars += len(doc_text)

        return context

    def search_by_project(
        self, project_name: str, query: Optional[str] = None, k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        プロジェクト名で検索（クエリはオプション）

        Args:
            project_name: プロジェクト名
            query: 検索クエリ（オプション）
            k: 返す結果数（デフォルト: 10）

        Returns:
            (Document, 類似度スコア)のタプルのリスト
        """
        if query:
            # クエリがある場合は類似度検索
            return self.search_similar_documents(
                query=query, project_name=project_name, k=k
            )
        else:
            # クエリがない場合はプロジェクト名でフィルタのみ
            # （仮のクエリを使用して全件取得的な動作）
            logger.info(f"プロジェクト '{project_name}' の全ドキュメントを取得")
            # プロジェクト名自体をクエリとして使用
            return self.search_similar_documents(
                query=project_name, project_name=project_name, k=k
            )

    def get_cross_project_insights(
        self, query: str, exclude_project: Optional[str] = None, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        他プロジェクトからの知見を取得

        Args:
            query: 検索クエリ
            exclude_project: 除外するプロジェクト名（現在のプロジェクト）
            k: 返す結果数（デフォルト: 5）

        Returns:
            (Document, 類似度スコア)のタプルのリスト
        """
        try:
            # クエリをベクトル化
            query_vector = self.embeddings.embed_query(query)

            # フィルタ条件の構築
            filter_dict = None
            if exclude_project:
                filter_dict = {"project_name": {"$ne": exclude_project}}
                logger.info(f"プロジェクト名でフィルタ: {exclude_project}")

            # 除外プロジェクトがある場合のフィルタ
            results = self.vector_store.similarity_search(
                query_vector=query_vector, k=k,filter_dict=filter_dict
            )

            # 除外プロジェクトをフィルタリング
            if exclude_project:
                filtered_results = [
                    (doc, score)
                    for doc, score in results
                    if doc.metadata.get("project_name") != exclude_project
                ]
                # k件に制限
                return filtered_results[:k]
            else:
                return results[:k]

        except Exception as e:
            logger.error(f"クロスプロジェクト検索でエラーが発生: {e}")
            return []

    def create_enhanced_prompt(
        self,
        base_prompt: str,
        query: str,
        project_name: Optional[str] = None,
        k: int = 5,
        max_context_chars: int = 5000,
    ) -> str:
        """
        RAGコンテキストを含む拡張プロンプトを作成

        Args:
            base_prompt: 基本となるプロンプト
            query: 検索クエリ
            project_name: プロジェクト名（オプション）
            k: 検索する関連ドキュメント数
            max_context_chars: コンテキストの最大文字数

        Returns:
            RAGコンテキストを含む拡張されたプロンプト
        """
        # 類似ドキュメントを検索
        results = self.search_similar_documents(
            query=query, project_name=project_name, k=k
        )

        if not results:
            # 検索結果がない場合は基本プロンプトをそのまま返す
            return base_prompt

        # 検索結果をフォーマット
        rag_context = self.format_context_for_prompt(
            results=results, max_chars=max_context_chars
        )

        # プロンプトにRAGコンテキストを統合
        enhanced_prompt = f"{base_prompt}\n\n"
        enhanced_prompt += "=" * 70 + "\n"
        enhanced_prompt += rag_context
        enhanced_prompt += "=" * 70 + "\n\n"
        enhanced_prompt += "上記の関連情報も考慮して、より深い分析を行ってください。\n"

        return enhanced_prompt

    # =========================================================================
    # 時系列重み付け機能
    # =========================================================================

    def _convert_distance_to_similarity(self, distance: float) -> float:
        """
        S3 Vectorsの距離スコアを類似度（0.0-1.0）に変換

        Args:
            distance: S3 Vectorsから返された距離スコア

        Returns:
            類似度スコア（0.0-1.0）
        """
        distance_metric = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")

        if distance_metric == "cosine":
            # cosine距離は 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - distance))
        elif distance_metric == "euclidean":
            # euclidean距離を類似度に変換
            similarity = 1.0 / (1.0 + distance)
        else:
            # フォールバック: 既存の計算方法（互換性のため）
            similarity = (1 + distance) / 2

        return similarity

    def _calculate_time_score(
        self, modified_at: Optional[str], created_at: Optional[str]
    ) -> float:
        """
        ドキュメントの時間スコアを計算（0.0-1.0）

        新しいドキュメントほど高いスコアを返します。
        modified_atを優先し、なければcreated_atを使用します。

        Args:
            modified_at: 更新日時（ISO 8601形式）
            created_at: 作成日時（ISO 8601形式）

        Returns:
            時間スコア（0.0-1.0）
        """
        # タイムスタンプがない場合は中間値を返す
        timestamp_str = modified_at or created_at
        if not timestamp_str:
            return 0.5

        try:
            # ISO 8601形式の日時をパース
            doc_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)

            # 経過日数を計算
            days_old = (now - doc_time).days

            # 新しいほど高スコア（指数関数的に減衰）
            # デフォルトは90日で約50%に減衰
            decay_days = float(os.getenv("RAG_DECAY_DAYS", "90"))
            time_score = math.exp(-days_old / decay_days)

            return max(0.0, min(1.0, time_score))

        except (ValueError, TypeError) as e:
            logger.warning(f"時間スコア計算でエラー: {e}, timestamp={timestamp_str}")
            return 0.5

    def _hybrid_scoring(
        self, results: List[Tuple[Document, float]], time_weight: float = 0.2
    ) -> List[Tuple[Document, float]]:
        """
        ハイブリッドスコアリング方式

        類似度スコアと時間スコアを重み付けして統合します。
        final_score = (similarity × (1-α)) + (time_score × α)

        Args:
            results: 検索結果（Document, 距離）のリスト
            time_weight: 時間の重み（0.0-1.0、デフォルト: 0.2）

        Returns:
            スコアリング済みの結果（Document, 統合スコア）のリスト
        """
        scored_results = []

        for doc, distance in results:
            # 距離を類似度に変換
            similarity = self._convert_distance_to_similarity(distance)

            # 時間スコアを計算
            time_score = self._calculate_time_score(
                doc.metadata.get("modified_at"), doc.metadata.get("created_at")
            )

            # ハイブリッドスコアを計算
            hybrid_score = (similarity * (1 - time_weight)) + (time_score * time_weight)

            scored_results.append((doc, hybrid_score))

        # スコアの降順でソート
        scored_results.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"ハイブリッドスコアリング完了（time_weight={time_weight}）")
        return scored_results

    def _reranking(
        self, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        リランキング方式

        類似度で検索した結果を時間でソートします。
        元の距離スコアは保持されます。

        Args:
            results: 検索結果（Document, 距離）のリスト

        Returns:
            時間でソート済みの結果（Document, 距離）のリスト
        """
        # 時間スコアでソート
        sorted_results = sorted(
            results,
            key=lambda x: self._calculate_time_score(
                x[0].metadata.get("modified_at"), x[0].metadata.get("created_at")
            ),
            reverse=True,
        )

        logger.info("リランキング完了（時間でソート）")
        return sorted_results

    def _time_decay(
        self, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        時間減衰方式

        類似度スコアに時間減衰を適用します。
        final_score = similarity × time_decay

        Args:
            results: 検索結果（Document, 距離）のリスト

        Returns:
            時間減衰適用済みの結果（Document, 減衰後スコア）のリスト
        """
        decayed_results = []

        for doc, distance in results:
            # 距離を類似度に変換
            similarity = self._convert_distance_to_similarity(distance)

            # 時間減衰係数を計算
            time_decay = self._calculate_time_score(
                doc.metadata.get("modified_at"), doc.metadata.get("created_at")
            )

            # 減衰適用済みスコアを計算
            decayed_score = similarity * time_decay

            decayed_results.append((doc, decayed_score))

        # スコアの降順でソート
        decayed_results.sort(key=lambda x: x[1], reverse=True)

        logger.info("時間減衰適用完了")
        return decayed_results

    def apply_time_series_weighting(
        self, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        環境変数に基づいて時系列重み付けを適用

        RAG_SCORING_METHOD環境変数によって適切な方式を選択します：
        - 'hybrid': ハイブリッドスコアリング（デフォルト）
        - 'reranking': リランキング方式
        - 'time_decay': 時間減衰方式
        - 'none': 重み付けなし（元の結果をそのまま返す）

        Args:
            results: 検索結果（Document, 距離/スコア）のリスト

        Returns:
            時系列重み付け適用済みの結果（Document, スコア）のリスト
        """
        if not results:
            return results

        # 環境変数から方式を取得
        scoring_method = os.getenv("RAG_SCORING_METHOD", "hybrid").lower()

        if scoring_method == "hybrid":
            # ハイブリッドスコアリング
            time_weight = float(os.getenv("RAG_TIME_WEIGHT", "0.2"))
            return self._hybrid_scoring(results, time_weight)

        elif scoring_method == "reranking":
            # リランキング
            return self._reranking(results)

        elif scoring_method == "time_decay":
            # 時間減衰
            return self._time_decay(results)

        elif scoring_method == "none":
            # 重み付けなし
            logger.info("時系列重み付けをスキップ")
            return results

        else:
            # 不明な方式の場合はデフォルト（ハイブリッド）を使用
            logger.warning(
                f"不明なスコアリング方式: {scoring_method}、ハイブリッドを使用"
            )
            time_weight = float(os.getenv("RAG_TIME_WEIGHT", "0.2"))
            return self._hybrid_scoring(results, time_weight)
