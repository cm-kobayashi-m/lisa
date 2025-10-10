"""
RAG検索を行うヘルパークラス

S3 Vectorsから類似ドキュメントを検索し、プロンプトに統合するための機能を提供します。
"""
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

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
        self,
        query: str,
        project_name: Optional[str] = None,
        file_name: Optional[str] = None,
        k: int = 5
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
            if project_name or file_name:
                filter_dict = {}
                if project_name:
                    filter_dict["project_name"] = {"$eq": project_name}
                    logger.info(f"プロジェクト名でフィルタ: {project_name}")
                if file_name:
                    filter_dict["file_name"] = {"$eq": file_name}
                    logger.info(f"ファイル名でフィルタ: {file_name}")

            # 類似度検索
            logger.info(f"S3 Vectorsで類似度検索を実行（k={k}）")
            results = self.vector_store.similarity_search(
                query_vector=query_vector,
                k=k,
                filter_dict=filter_dict
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
        include_metadata: bool = True
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
        context += f"以下は過去の類似プロジェクトや関連ドキュメントから抽出された情報です。\n"
        context += f"これらの情報を参考にして、より深い分析を行ってください。\n\n"

        total_chars = len(context)

        for i, (doc, score) in enumerate(results, 1):
            # コサイン類似度を0-100%の範囲に変換
            # コサイン類似度は-1から1の範囲なので、0から1に正規化してから%に変換
            similarity_percent = (1 + score) / 2 * 100

            # ドキュメントのテキストをフォーマット
            doc_text = f"### {i}. 関連度: {similarity_percent:.1f}%\n"

            if include_metadata:
                # メタデータを含める
                doc_text += f"- **プロジェクト**: {doc.metadata.get('project_name', '不明')}\n"
                doc_text += f"- **ファイル**: {doc.metadata.get('file_name', '不明')}\n"

                # その他のメタデータ（あれば）
                if 'chunk_index' in doc.metadata:
                    doc_text += f"- **チャンク番号**: {doc.metadata['chunk_index']}\n"
                if 'title' in doc.metadata:
                    doc_text += f"- **タイトル**: {doc.metadata['title']}\n"
                if 'topics' in doc.metadata and doc.metadata['topics']:
                    doc_text += f"- **トピック**: {', '.join(doc.metadata['topics'])}\n"
                if 'importance' in doc.metadata:
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
                context += f"\n（以降、文字数制限により省略）\n"
                break

            # 文字数チェック
            if total_chars + len(doc_text) > max_chars:
                context += f"\n（文字数制限により、残り{len(results) - i + 1}件の結果を省略）\n"
                break

            context += doc_text
            total_chars += len(doc_text)

        return context

    def search_by_project(
        self,
        project_name: str,
        query: Optional[str] = None,
        k: int = 10
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
                query=query,
                project_name=project_name,
                k=k
            )
        else:
            # クエリがない場合はプロジェクト名でフィルタのみ
            # （仮のクエリを使用して全件取得的な動作）
            logger.info(f"プロジェクト '{project_name}' の全ドキュメントを取得")
            # プロジェクト名自体をクエリとして使用
            return self.search_similar_documents(
                query=project_name,
                project_name=project_name,
                k=k
            )

    def get_cross_project_insights(
        self,
        query: str,
        exclude_project: Optional[str] = None,
        k: int = 5
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

            # 除外プロジェクトがある場合のフィルタ
            # S3 Vectorsは$ne（not equal）をサポートしていない可能性があるため、
            # 全結果を取得してからPython側でフィルタリング
            results = self.vector_store.similarity_search(
                query_vector=query_vector,
                k=k * 2  # 除外後の結果が少なくなることを考慮して多めに取得
            )

            # 除外プロジェクトをフィルタリング
            if exclude_project:
                filtered_results = [
                    (doc, score) for doc, score in results
                    if doc.metadata.get('project_name') != exclude_project
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
        max_context_chars: int = 5000
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
            query=query,
            project_name=project_name,
            k=k
        )

        if not results:
            # 検索結果がない場合は基本プロンプトをそのまま返す
            return base_prompt

        # 検索結果をフォーマット
        rag_context = self.format_context_for_prompt(
            results=results,
            max_chars=max_context_chars
        )

        # プロンプトにRAGコンテキストを統合
        enhanced_prompt = f"{base_prompt}\n\n"
        enhanced_prompt += "=" * 70 + "\n"
        enhanced_prompt += rag_context
        enhanced_prompt += "=" * 70 + "\n\n"
        enhanced_prompt += "上記の関連情報も考慮して、より深い分析を行ってください。\n"

        return enhanced_prompt