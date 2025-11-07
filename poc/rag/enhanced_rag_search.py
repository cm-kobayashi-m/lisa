#!/usr/bin/env python3
"""
Enhanced RAG Search - CRAG統合検索

generate_note.pyから呼び出される統合インターフェース。
既存のRAG-FusionとCRAG（関連性評価＋Knowledge Refinement）を統合。
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .corrective_rag import InternalCRAG, create_internal_crag, CRAGSearchResult
from .evaluator import RelevanceLevel


@dataclass
class EnhancedRAGConfig:
    """Enhanced RAGの設定"""
    # CRAG設定
    use_crag: bool = True                    # CRAGを有効にするか
    crag_upper_threshold: float = 0.5        # CORRECT判定閾値
    crag_lower_threshold: float = -0.5       # INCORRECT判定閾値
    use_knowledge_refinement: bool = True    # Knowledge Refinementを有効にするか
    max_refined_segments: int = 5            # 最大精製セグメント数

    # 既存のRAG設定（互換性のため）
    use_rag_fusion: bool = True              # RAG-Fusionを有効にするか
    num_fusion_queries: int = 3              # Fusion用クエリ数
    use_hybrid_scoring: bool = True          # ハイブリッドスコアリングを有効にするか
    time_weight: float = 0.2                 # 時系列重み

    # 共通設定
    min_score: float = 0.3                   # 最小類似度スコア
    max_total_docs: int = 30                 # 最大取得文書数


class EnhancedRAGSearch:
    """
    Enhanced RAG検索クラス

    CRAGとRAG-Fusionを統合し、より精度の高い検索を実現
    """

    def __init__(
        self,
        retriever,
        embeddings,
        gemini_client,
        config: Optional[EnhancedRAGConfig] = None
    ):
        """
        初期化

        Args:
            retriever: RAGRetriever インスタンス
            embeddings: 埋め込みモデル
            gemini_client: Gemini APIクライアント
            config: Enhanced RAG設定
        """
        self.retriever = retriever
        self.embeddings = embeddings
        self.gemini_client = gemini_client
        self.config = config or EnhancedRAGConfig()

        # CRAG初期化
        if self.config.use_crag:
            self.crag = create_internal_crag(
                retriever=retriever,
                embeddings=embeddings,
                upper_threshold=self.config.crag_upper_threshold,
                lower_threshold=self.config.crag_lower_threshold
            )
        else:
            self.crag = None

    def search_with_enhancements(
        self,
        query: str,
        project_name: str,
        k_current: int = 10,
        k_similar: int = 10
    ) -> Dict[str, Any]:
        """
        拡張検索を実行

        Args:
            query: 検索クエリ
            project_name: プロジェクト名
            k_current: 現在のプロジェクトから取得する件数
            k_similar: 類似プロジェクトから取得する件数

        Returns:
            検索結果の辞書
        """
        results = {
            "current_project_results": [],
            "similar_project_results": [],
            "relevance_level": None,
            "strategy_used": None,
            "refined_documents": [],
            "confidence": 0.0,
            "rag_context": ""
        }

        # CRAGを使用する場合
        if self.config.use_crag:
            print(f"[Enhanced RAG] CRAG検索を実行中...")
            crag_result = self._execute_crag_search(
                query, project_name, k_current
            )

            # 結果を整理
            results["current_project_results"] = crag_result.documents[:k_current]
            results["relevance_level"] = crag_result.relevance_level.value
            results["strategy_used"] = crag_result.strategy_used
            results["refined_documents"] = crag_result.refined_documents
            results["confidence"] = crag_result.total_confidence

            # 関連性レベルに応じて類似プロジェクト検索
            if crag_result.relevance_level == RelevanceLevel.INCORRECT:
                # 低関連性の場合は類似プロジェクトを重視
                print(f"[Enhanced RAG] 低関連性のため類似プロジェクトを追加検索...")
                similar_results = self._search_similar_projects_enhanced(
                    query, project_name, k_similar * 2  # より多く取得
                )
                results["similar_project_results"] = similar_results

            elif crag_result.relevance_level == RelevanceLevel.AMBIGUOUS:
                # 中間の場合は通常の類似プロジェクト検索
                print(f"[Enhanced RAG] 中間関連性のため類似プロジェクトを追加検索...")
                similar_results = self._search_similar_projects_enhanced(
                    query, project_name, k_similar
                )
                results["similar_project_results"] = similar_results

            # else CORRECT: 類似プロジェクトは最小限

        else:
            # 従来のRAG検索（CRAG無効時のフォールバック）
            print(f"[Enhanced RAG] 従来のRAG検索を実行中...")
            results = self._execute_legacy_search(
                query, project_name, k_current, k_similar
            )

        # RAGコンテキストを構築
        results["rag_context"] = self._build_enhanced_context(results)

        return results

    def _execute_crag_search(
        self,
        query: str,
        project_name: str,
        k: int
    ) -> CRAGSearchResult:
        """
        CRAG検索を実行

        Args:
            query: 検索クエリ
            project_name: プロジェクト名
            k: 取得する文書数

        Returns:
            CRAG検索結果
        """
        return self.crag.search(
            query=query,
            project_name=project_name,
            k=k,
            min_score=self.config.min_score,
            use_refinement=self.config.use_knowledge_refinement
        )

    def _search_similar_projects_enhanced(
        self,
        query: str,
        exclude_project: str,
        k: int
    ) -> List[Tuple[Any, float]]:
        """
        類似プロジェクトの拡張検索

        Args:
            query: 検索クエリ
            exclude_project: 除外するプロジェクト名
            k: 取得する文書数

        Returns:
            検索結果リスト
        """
        try:
            # RAG-Fusionが有効な場合
            if self.config.use_rag_fusion:
                from generate_note import generate_multiple_queries, multi_query_search

                # 複数クエリ生成
                queries = generate_multiple_queries(
                    self.gemini_client,
                    exclude_project,
                    base_context=query,
                    num_queries=self.config.num_fusion_queries
                )

                # 並行検索
                results = multi_query_search(
                    self.retriever,
                    queries,
                    None,  # プロジェクト制限なし
                    k,
                    self.config.min_score,
                    use_rrf=True
                )
            else:
                # 通常の類似プロジェクト検索
                results = self.retriever.get_cross_project_insights(
                    query=query,
                    exclude_project=exclude_project,
                    k=k
                )

            return results

        except Exception as e:
            print(f"[Enhanced RAG] 類似プロジェクト検索エラー: {e}")
            return []

    def _execute_legacy_search(
        self,
        query: str,
        project_name: str,
        k_current: int,
        k_similar: int
    ) -> Dict[str, Any]:
        """
        従来のRAG検索を実行（フォールバック）

        Args:
            query: 検索クエリ
            project_name: プロジェクト名
            k_current: 現在のプロジェクトから取得する件数
            k_similar: 類似プロジェクトから取得する件数

        Returns:
            検索結果の辞書
        """
        results = {
            "current_project_results": [],
            "similar_project_results": [],
            "relevance_level": "unknown",
            "strategy_used": "legacy_rag",
            "refined_documents": [],
            "confidence": 0.5,
            "rag_context": ""
        }

        # 現在のプロジェクト検索
        try:
            current_results = self.retriever.search_similar_documents(
                query=query,
                project_name=project_name,
                k=k_current
            )
            results["current_project_results"] = current_results
        except Exception as e:
            print(f"[Enhanced RAG] 現在のプロジェクト検索エラー: {e}")

        # 類似プロジェクト検索
        try:
            similar_results = self.retriever.get_cross_project_insights(
                query=query,
                exclude_project=project_name,
                k=k_similar
            )
            results["similar_project_results"] = similar_results
        except Exception as e:
            print(f"[Enhanced RAG] 類似プロジェクト検索エラー: {e}")

        return results

    def _build_enhanced_context(self, results: Dict[str, Any]) -> str:
        """
        拡張コンテキストを構築

        Args:
            results: 検索結果

        Returns:
            フォーマット済みのRAGコンテキスト
        """
        context_parts = []

        # 関連性レベルの情報を追加
        if results.get("relevance_level"):
            confidence = results.get("confidence", 0.0)
            context_parts.append(f"## 検索品質評価\n")
            context_parts.append(f"- 関連性レベル: {results['relevance_level']}\n")
            context_parts.append(f"- 確信度: {confidence:.2%}\n")
            context_parts.append(f"- 使用戦略: {results.get('strategy_used', 'N/A')}\n")
            context_parts.append("\n")

        # 精製されたドキュメントがある場合
        if results.get("refined_documents"):
            context_parts.append("## 【精製された重要情報】\n\n")
            for i, refined in enumerate(results["refined_documents"][:3], 1):
                context_parts.append(f"### 重要ドキュメント {i}\n")
                context_parts.append(f"{refined.refined_content[:1000]}\n")
                context_parts.append(f"（選択率: {refined.selected_segments}/{refined.total_segments}セグメント）\n\n")

        # 現在のプロジェクト情報
        if results.get("current_project_results"):
            context_parts.append("\n## 【現在のプロジェクトの情報】\n\n")
            context_parts.append(self._format_search_results(
                results["current_project_results"][:5]
            ))

        # 類似プロジェクト情報
        if results.get("similar_project_results"):
            context_parts.append("\n## 【類似プロジェクトからの参考情報】\n\n")
            context_parts.append(self._format_search_results(
                results["similar_project_results"][:5]
            ))

        return ''.join(context_parts)

    def _format_search_results(
        self,
        results: List[Tuple[Any, float]],
        max_chars: int = 500
    ) -> str:
        """
        検索結果をフォーマット

        Args:
            results: 検索結果リスト
            max_chars: 各ドキュメントの最大文字数

        Returns:
            フォーマット済みテキスト
        """
        formatted = []

        for i, (doc, distance) in enumerate(results, 1):
            doc_text = doc.text if hasattr(doc, 'text') else str(doc)
            doc_text = doc_text[:max_chars] + "..." if len(doc_text) > max_chars else doc_text

            # メタデータ情報
            metadata_info = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'project' in doc.metadata:
                    metadata_info += f"[プロジェクト: {doc.metadata['project']}] "
                if 'doc_type' in doc.metadata:
                    metadata_info += f"[タイプ: {doc.metadata['doc_type']}] "

            formatted.append(f"**[{i}] {metadata_info}**\n{doc_text}\n")

        return '\n'.join(formatted)


def create_enhanced_rag_search(
    retriever,
    embeddings,
    gemini_client,
    config: Optional[EnhancedRAGConfig] = None
) -> EnhancedRAGSearch:
    """
    Enhanced RAG検索のファクトリー関数

    Args:
        retriever: RAGRetriever インスタンス
        embeddings: 埋め込みモデル
        gemini_client: Gemini APIクライアント
        config: 設定

    Returns:
        EnhancedRAGSearchインスタンス
    """
    return EnhancedRAGSearch(retriever, embeddings, gemini_client, config)


def integrate_with_generate_note(
    gemini_client,
    project_name: str,
    enable_crag: bool = True
) -> Tuple[str, str]:
    """
    generate_note.pyとの統合関数

    既存のgenerate_note.pyから呼び出される統合ポイント。
    この関数を generate_final_reflection_note_v2 の代わりに使用可能。

    Args:
        gemini_client: Gemini APIクライアント
        project_name: プロジェクト名
        enable_crag: CRAGを有効にするか

    Returns:
        (リフレクションノート, RAGコンテキスト)
    """
    from rag.rag_retriever import RAGRetriever
    from rag.vector_store import S3VectorStore
    from rag.embeddings import GeminiEmbeddings
    import os

    # 初期化
    api_key = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', "gemini-embedding-001")
    DIMENSION = int(os.getenv('DIMENSION', 1536))

    embeddings = GeminiEmbeddings(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
        dimension=DIMENSION
    )

    vector_store = S3VectorStore(
        vector_bucket_name=os.getenv('VECTOR_BUCKET_NAME', 'lisa-poc-vectors'),
        index_name=os.getenv('VECTOR_INDEX_NAME', 'project-documents'),
        dimension=DIMENSION,
        region_name=os.getenv('AWS_REGION', 'us-west-2'),
        create_if_not_exists=False
    )

    retriever = RAGRetriever(vector_store, embeddings)

    # Enhanced RAG設定
    config = EnhancedRAGConfig(
        use_crag=enable_crag,
        use_rag_fusion=os.getenv('USE_RAG_FUSION', 'true').lower() == 'true',
        use_knowledge_refinement=True,
        min_score=float(os.getenv('RAG_MIN_SCORE', '0.3'))
    )

    # Enhanced RAG検索を実行
    enhanced_search = create_enhanced_rag_search(
        retriever, embeddings, gemini_client, config
    )

    # プロジェクト名からキーワード生成（既存の関数を使用）
    from generate_note import generate_project_keywords
    try:
        keywords = generate_project_keywords(gemini_client, project_name)
    except:
        keywords = project_name

    # 拡張検索を実行
    results = enhanced_search.search_with_enhancements(
        query=keywords,
        project_name=project_name,
        k_current=int(os.getenv('RAG_ONLY_MODE_K_CURRENT', '15')),
        k_similar=int(os.getenv('RAG_ONLY_MODE_K_SIMILAR', '15'))
    )

    # リフレクションノートを生成（既存の関数を使用）
    from improved_prompts import get_final_reflection_prompt

    prompt = get_final_reflection_prompt(
        project_name=project_name,
        summaries_text=results["rag_context"]
    )

    # 関連性レベルに応じてプロンプトを調整
    if results.get("relevance_level") == "incorrect":
        prompt = f"""【注意】検索結果の関連性が低いため、利用可能な情報が限定的です。
可能な範囲で推測を交えながらリフレクションノートを生成してください。

{prompt}"""

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro ')

    try:
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text, results["rag_context"]
    except Exception as e:
        print(f"[ERROR] ノート生成エラー: {e}")
        raise
