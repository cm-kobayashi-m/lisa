"""
RAGリトリーバー

ベクトルストアから関連情報を検索し、
Gemini分析用のコンテキストを生成します。
"""
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .vector_store import S3VectorStore, Document
from .embeddings import GeminiEmbeddings

# ロギング設定
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """検索結果を表すデータクラス"""
    documents: List[Document]  # 検索されたドキュメント
    scores: List[float]  # 各ドキュメントのスコア（距離/類似度）
    context: str  # 生成されたコンテキスト文字列
    metadata: Dict[str, Any]  # 追加メタデータ


class RAGRetriever:
    """
    RAG用のリトリーバー

    ベクトルストアから関連ドキュメントを検索し、
    LLMに渡すためのコンテキストを生成します。
    """

    def __init__(
        self,
        vector_store: S3VectorStore,
        embeddings: GeminiEmbeddings,
        top_k: int = 5,
        score_threshold: float = 0.7
    ):
        """
        リトリーバーを初期化

        Args:
            vector_store: S3Vectorsストア
            embeddings: Gemini Embeddings
            top_k: 検索結果の最大数
            score_threshold: 最小類似度スコア
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.top_k = min(top_k, 30)  # S3 Vectorsの制限
        self.score_threshold = score_threshold

        logger.info(f"RAGリトリーバー初期化: top_k={self.top_k}")

    def get_relevant_context(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        rerank: bool = False
    ) -> RetrievalResult:
        """
        クエリに関連するコンテキストを取得

        Args:
            query: 検索クエリ
            k: 返す結果の数（Noneの場合はデフォルト値）
            filter_dict: フィルタ条件
            rerank: リランキングを行うか

        Returns:
            RetrievalResult オブジェクト
        """
        if not query or not query.strip():
            logger.warning("空のクエリが渡されました")
            return RetrievalResult(
                documents=[],
                scores=[],
                context="",
                metadata={}
            )

        k = k or self.top_k

        try:
            # クエリをベクトル化
            logger.info(f"クエリをベクトル化中: '{query[:50]}...'")
            query_vector = self.embeddings.embed_query(query)

            # ベクトル検索を実行
            logger.info(f"ベクトル検索実行中 (k={k})")
            search_results = self.vector_store.similarity_search(
                query_vector=query_vector,
                k=k,
                filter_dict=filter_dict
            )

            if not search_results:
                logger.info("検索結果が見つかりませんでした")
                return RetrievalResult(
                    documents=[],
                    scores=[],
                    context="",
                    metadata={'query': query, 'filter': filter_dict}
                )

            # 結果を分離
            documents = []
            scores = []
            for doc, score in search_results:
                # スコアフィルタリング（コサイン距離が小さいほど類似）
                if score <= (1.0 - self.score_threshold):
                    documents.append(doc)
                    scores.append(score)

            logger.info(f"{len(documents)}件の関連ドキュメントを取得")

            # リランキング（必要に応じて）
            if rerank and len(documents) > 1:
                documents, scores = self._rerank_documents(
                    query, documents, scores
                )

            # コンテキストを生成
            context = self.format_context_for_prompt(documents, scores)

            # メタデータを収集
            metadata = {
                'query': query,
                'filter': filter_dict,
                'total_results': len(documents),
                'avg_score': sum(scores) / len(scores) if scores else 0
            }

            return RetrievalResult(
                documents=documents,
                scores=scores,
                context=context,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"コンテキスト取得中にエラー: {e}")
            return RetrievalResult(
                documents=[],
                scores=[],
                context="",
                metadata={'error': str(e)}
            )

    def format_context_for_prompt(
        self,
        documents: List[Document],
        scores: Optional[List[float]] = None,
        max_length: int = 10000
    ) -> str:
        """
        プロンプト用にコンテキストをフォーマット

        Args:
            documents: ドキュメントのリスト
            scores: スコアのリスト
            max_length: 最大文字数

        Returns:
            フォーマットされたコンテキスト文字列
        """
        if not documents:
            return ""

        context_parts = []
        total_length = 0

        context_parts.append("## 関連ドキュメントからの参考情報:\n")

        for i, doc in enumerate(documents):
            # スコアがある場合は追加
            score_str = ""
            if scores and i < len(scores):
                # コサイン距離を類似度に変換（1 - distance）
                similarity = 1.0 - scores[i]
                score_str = f" (類似度: {similarity:.2%})"

            # ドキュメント情報をフォーマット
            doc_header = f"\n### [{i+1}] {doc.metadata.get('file_name', 'Unknown')}{score_str}"
            doc_info = f"プロジェクト: {doc.metadata.get('project_name', 'Unknown')}\n"

            # チャンク情報
            chunk_info = ""
            if doc.metadata.get('chunk_index') is not None:
                chunk_info = f"チャンク: {doc.metadata['chunk_index'] + 1}/{doc.metadata.get('total_chunks', '?')}\n"

            # テキスト本文
            doc_text = f"\n{doc.text}\n"

            # 区切り線
            separator = "\n" + "-" * 50 + "\n"

            # 長さチェック
            part = doc_header + doc_info + chunk_info + doc_text + separator
            part_length = len(part)

            if total_length + part_length > max_length:
                # 最大長を超える場合は切り詰め
                remaining = max_length - total_length
                if remaining > 100:  # 最低100文字は含める
                    truncated_text = doc.text[:remaining - len(doc_header) - len(doc_info) - 20]
                    part = doc_header + doc_info + chunk_info + f"\n{truncated_text}... [truncated]\n"
                    context_parts.append(part)
                break

            context_parts.append(part)
            total_length += part_length

        # 統計情報を追加
        stats = f"\n## 検索統計:\n"
        stats += f"- 検索結果数: {len(documents)}件\n"
        if scores:
            avg_similarity = 1.0 - (sum(scores) / len(scores))
            stats += f"- 平均類似度: {avg_similarity:.2%}\n"

        context_parts.append(stats)

        return "".join(context_parts)

    def _rerank_documents(
        self,
        query: str,
        documents: List[Document],
        scores: List[float]
    ) -> Tuple[List[Document], List[float]]:
        """
        ドキュメントをリランキング（詳細な類似度計算）

        Args:
            query: 検索クエリ
            documents: ドキュメントリスト
            scores: 初期スコアリスト

        Returns:
            リランキング後の(ドキュメント, スコア)
        """
        # 簡単なリランキング: クエリとの単語重複率を考慮
        query_words = set(query.lower().split())

        reranked = []
        for doc, score in zip(documents, scores):
            doc_words = set(doc.text.lower().split())
            word_overlap = len(query_words & doc_words) / len(query_words) if query_words else 0

            # 元のスコアと単語重複率を組み合わせ
            combined_score = score * 0.7 + (1.0 - word_overlap) * 0.3
            reranked.append((doc, combined_score))

        # スコアでソート（昇順: 距離が小さい方が良い）
        reranked.sort(key=lambda x: x[1])

        # 分離して返す
        reranked_docs = [item[0] for item in reranked]
        reranked_scores = [item[1] for item in reranked]

        return reranked_docs, reranked_scores

    def get_similar_projects(
        self,
        project_name: str,
        k: int = 3
    ) -> List[str]:
        """
        類似プロジェクトを取得

        Args:
            project_name: 基準となるプロジェクト名
            k: 返すプロジェクト数

        Returns:
            類似プロジェクト名のリスト
        """
        # プロジェクトのドキュメントを検索
        filter_dict = {"project_name": {"$eq": project_name}}

        # サンプルドキュメントを取得
        sample_keys, _ = self.vector_store.list_documents(max_results=1)
        if not sample_keys:
            return []

        sample_doc = self.vector_store.get_document(sample_keys[0])
        if not sample_doc or not sample_doc.vector:
            return []

        # 類似検索（自プロジェクトを除外）
        results = self.vector_store.similarity_search(
            query_vector=sample_doc.vector,
            k=k + 5,  # 多めに取得して自プロジェクトを除外
            filter_dict={"project_name": {"$ne": project_name}}
        )

        # プロジェクト名を収集（重複排除）
        similar_projects = []
        seen = set()
        for doc, _ in results:
            proj_name = doc.metadata.get('project_name', '')
            if proj_name and proj_name not in seen:
                similar_projects.append(proj_name)
                seen.add(proj_name)
                if len(similar_projects) >= k:
                    break

        return similar_projects

    def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        k: int = 10
    ) -> List[Document]:
        """
        メタデータで検索

        Args:
            metadata_filters: メタデータフィルタ
            k: 返す結果数

        Returns:
            ドキュメントのリスト
        """
        # ダミーベクトルで検索（メタデータフィルタのみ使用）
        dummy_vector = [0.0] * self.embeddings.dimension

        results = self.vector_store.similarity_search(
            query_vector=dummy_vector,
            k=k,
            filter_dict=metadata_filters
        )

        return [doc for doc, _ in results]

    def get_project_summary(
        self,
        project_name: str
    ) -> Dict[str, Any]:
        """
        プロジェクトの要約情報を取得

        Args:
            project_name: プロジェクト名

        Returns:
            要約情報の辞書
        """
        # プロジェクトのドキュメントを取得
        docs = self.search_by_metadata(
            metadata_filters={"project_name": {"$eq": project_name}},
            k=100
        )

        if not docs:
            return {
                'project_name': project_name,
                'document_count': 0,
                'file_count': 0,
                'doc_types': []
            }

        # 統計情報を集計
        file_names = set()
        doc_types = set()
        total_chunks = 0

        for doc in docs:
            file_names.add(doc.metadata.get('file_name', ''))
            doc_types.add(doc.metadata.get('doc_type', ''))
            total_chunks += 1

        return {
            'project_name': project_name,
            'document_count': total_chunks,
            'file_count': len(file_names),
            'files': list(file_names),
            'doc_types': list(doc_types)
        }