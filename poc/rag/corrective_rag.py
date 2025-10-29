#!/usr/bin/env python3
"""
Corrective RAG (CRAG) - 社内データ特化版の実装

検索結果の評価に基づいて、適切な検索戦略を選択し、
Knowledge Refinementで精製した情報を提供する。
"""

import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .evaluator import RelevanceEvaluator, RelevanceLevel, create_evaluator
from .knowledge_refiner import KnowledgeRefiner, RefinedDocument, create_refiner


@dataclass
class SearchStrategy:
    """検索戦略の定義"""
    name: str                       # 戦略名
    description: str                # 説明
    query_modifier: callable        # クエリ修正関数
    metadata_filter: Dict          # メタデータフィルタ
    priority: int                   # 優先度（低いほど高優先）


@dataclass
class CRAGSearchResult:
    """CRAG検索結果"""
    documents: List[Tuple[Any, float]]  # (ドキュメント, 距離)のリスト
    relevance_level: RelevanceLevel     # 全体の関連性レベル
    strategy_used: str                  # 使用された戦略
    refined_documents: List[RefinedDocument]  # 精製されたドキュメント
    total_confidence: float             # 全体の確信度


class InternalCRAG:
    """
    社内データ特化型Corrective RAG

    外部Web検索の代わりに、社内データの多層的な検索戦略を使用
    """

    def __init__(
        self,
        retriever,
        evaluator: Optional[RelevanceEvaluator] = None,
        refiner: Optional[KnowledgeRefiner] = None,
        embeddings=None,
        upper_threshold: float = 0.5,
        lower_threshold: float = -0.5
    ):
        """
        初期化

        Args:
            retriever: RAGRetriever インスタンス
            evaluator: 関連性評価器
            refiner: Knowledge Refiner
            embeddings: 埋め込みモデル
            upper_threshold: CORRECT判定の閾値
            lower_threshold: INCORRECT判定の閾値
        """
        self.retriever = retriever
        self.evaluator = evaluator or create_evaluator(
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold
        )
        self.refiner = refiner or create_refiner()
        self.embeddings = embeddings

        # 検索戦略の定義
        self._initialize_search_strategies()

    def _initialize_search_strategies(self):
        """検索戦略を初期化"""
        self.search_strategies = {
            "detailed_same_project": SearchStrategy(
                name="同一プロジェクト詳細検索",
                description="同じプロジェクト内でより詳細な検索",
                query_modifier=lambda q: f"{q} 詳細 実装 具体的",
                metadata_filter={"scope": "same_project"},
                priority=1
            ),
            "similar_industry": SearchStrategy(
                name="同業界類似プロジェクト検索",
                description="同じ業界の他プロジェクトから検索",
                query_modifier=lambda q: f"{q} 業界 事例 パターン",
                metadata_filter={"scope": "same_industry"},
                priority=2
            ),
            "same_tech_stack": SearchStrategy(
                name="同一技術スタック検索",
                description="同じ技術を使用したプロジェクトから検索",
                query_modifier=lambda q: f"{q} 技術 実装 アーキテクチャ",
                metadata_filter={"scope": "same_tech"},
                priority=3
            ),
            "abstract_concepts": SearchStrategy(
                name="抽象概念検索",
                description="より抽象的な概念で全体から検索",
                query_modifier=self._abstract_query,
                metadata_filter={"scope": "all"},
                priority=4
            ),
            "temporal_expansion": SearchStrategy(
                name="時期拡張検索",
                description="より広い時期範囲で検索",
                query_modifier=lambda q: q,
                metadata_filter={"time_range": "expanded"},
                priority=5
            ),
            "document_type_variation": SearchStrategy(
                name="ドキュメントタイプ変更検索",
                description="異なるドキュメントタイプで検索",
                query_modifier=lambda q: q,
                metadata_filter={"doc_type": "varied"},
                priority=6
            )
        }

    def search(
        self,
        query: str,
        project_name: str,
        k: int = 10,
        min_score: float = 0.3,
        use_refinement: bool = True
    ) -> CRAGSearchResult:
        """
        CRAG検索を実行

        Args:
            query: 検索クエリ
            project_name: プロジェクト名
            k: 取得する文書数
            min_score: 最小スコア
            use_refinement: Knowledge Refinementを使用するか

        Returns:
            CRAG検索結果
        """
        # Step 1: 初回検索
        print(f"[CRAG] 初回検索実行中...")
        initial_results = self.retriever.search_similar_documents(
            query=query,
            project_name=project_name,
            k=k
        )

        # Step 2: 検索結果を評価
        print(f"[CRAG] 検索結果を評価中...")
        evaluation_results, overall_level = self._evaluate_results(
            query, initial_results
        )

        print(f"[CRAG] 評価結果: {overall_level.value}")

        # Step 3: レベルに応じた処理
        if overall_level == RelevanceLevel.CORRECT:
            # 高関連性: そのまま使用（精製のみ）
            result = self._handle_correct(
                query, initial_results, project_name, use_refinement
            )

        elif overall_level == RelevanceLevel.INCORRECT:
            # 低関連性: 代替戦略を実行
            result = self._handle_incorrect(
                query, initial_results, project_name, k, min_score, use_refinement
            )

        else:  # AMBIGUOUS
            # 中間: 複数戦略を組み合わせ
            result = self._handle_ambiguous(
                query, initial_results, project_name, k, min_score, use_refinement
            )

        return result

    def _evaluate_results(
        self,
        query: str,
        results: List[Tuple[Any, float]]
    ) -> Tuple[List, RelevanceLevel]:
        """
        検索結果を評価

        Args:
            query: 検索クエリ
            results: 検索結果

        Returns:
            (評価結果リスト, 全体レベル)
        """
        documents = []
        for doc, distance in results:
            # ドキュメント情報を準備
            doc_content = doc.text if hasattr(doc, 'text') else str(doc)
            doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            documents.append((doc_content, doc_metadata))

        return self.evaluator.evaluate_batch(query, documents)

    def _handle_correct(
        self,
        query: str,
        results: List[Tuple[Any, float]],
        project_name: str,
        use_refinement: bool
    ) -> CRAGSearchResult:
        """
        CORRECTケースの処理

        Args:
            query: 検索クエリ
            results: 初回検索結果
            project_name: プロジェクト名
            use_refinement: 精製を使用するか

        Returns:
            検索結果
        """
        print(f"[CRAG] CORRECT: 検索結果を精製中...")

        refined_docs = []
        if use_refinement:
            for doc, distance in results[:5]:  # 上位5件を精製
                doc_content = doc.text if hasattr(doc, 'text') else str(doc)
                refined = self.refiner.refine(
                    document=doc_content,
                    query=query
                )
                refined_docs.append(refined)

        # 確信度計算
        confidence = self._calculate_overall_confidence(results, refined_docs)

        return CRAGSearchResult(
            documents=results,
            relevance_level=RelevanceLevel.CORRECT,
            strategy_used="initial_search",
            refined_documents=refined_docs,
            total_confidence=confidence
        )

    def _handle_incorrect(
        self,
        query: str,
        initial_results: List[Tuple[Any, float]],
        project_name: str,
        k: int,
        min_score: float,
        use_refinement: bool
    ) -> CRAGSearchResult:
        """
        INCORRECTケースの処理

        Args:
            query: 検索クエリ
            initial_results: 初回検索結果
            project_name: プロジェクト名
            k: 取得する文書数
            min_score: 最小スコア
            use_refinement: 精製を使用するか

        Returns:
            検索結果
        """
        print(f"[CRAG] INCORRECT: 代替検索戦略を実行中...")

        # 失敗パターンを分析
        failure_patterns = self._analyze_failure_patterns(initial_results)

        # 適切な戦略を選択
        selected_strategies = self._select_strategies(failure_patterns)

        best_result = None
        best_score = -1.0

        for strategy in selected_strategies:
            print(f"[CRAG] 戦略実行: {strategy.name}")

            # 修正されたクエリで検索
            modified_query = strategy.query_modifier(query)
            strategy_results = self._execute_strategy_search(
                modified_query, project_name, k, strategy
            )

            if not strategy_results:
                continue

            # 評価
            eval_results, level = self._evaluate_results(modified_query, strategy_results)

            # 最良の結果を保持
            if level != RelevanceLevel.INCORRECT:
                avg_score = sum(r.score for r in eval_results) / len(eval_results)
                if avg_score > best_score:
                    best_score = avg_score
                    best_result = (strategy_results, strategy.name, level)

        # 最良の結果を使用
        if best_result:
            results, strategy_name, level = best_result
            print(f"[CRAG] 最適戦略: {strategy_name}")
        else:
            # すべて失敗した場合は初回結果を使用
            print(f"[CRAG] 代替戦略も失敗、初回結果を使用")
            results = initial_results
            strategy_name = "fallback_to_initial"
            level = RelevanceLevel.INCORRECT

        # 精製処理
        refined_docs = []
        if use_refinement and results:
            for doc, distance in results[:3]:  # 上位3件を精製
                doc_content = doc.text if hasattr(doc, 'text') else str(doc)
                refined = self.refiner.refine(
                    document=doc_content,
                    query=query
                )
                refined_docs.append(refined)

        confidence = self._calculate_overall_confidence(results, refined_docs)

        return CRAGSearchResult(
            documents=results,
            relevance_level=level,
            strategy_used=strategy_name,
            refined_documents=refined_docs,
            total_confidence=confidence
        )

    def _handle_ambiguous(
        self,
        query: str,
        initial_results: List[Tuple[Any, float]],
        project_name: str,
        k: int,
        min_score: float,
        use_refinement: bool
    ) -> CRAGSearchResult:
        """
        AMBIGUOUSケースの処理

        Args:
            query: 検索クエリ
            initial_results: 初回検索結果
            project_name: プロジェクト名
            k: 取得する文書数
            min_score: 最小スコア
            use_refinement: 精製を使用するか

        Returns:
            検索結果
        """
        print(f"[CRAG] AMBIGUOUS: 複数戦略を組み合わせ中...")

        # 初回結果を含める
        combined_results = list(initial_results[:k//2])

        # 補完的な戦略を実行
        complementary_strategies = [
            self.search_strategies["similar_industry"],
            self.search_strategies["same_tech_stack"]
        ]

        for strategy in complementary_strategies:
            modified_query = strategy.query_modifier(query)
            strategy_results = self._execute_strategy_search(
                modified_query, project_name, k//2, strategy
            )
            if strategy_results:
                combined_results.extend(strategy_results)

        # 重複除去
        seen_keys = set()
        unique_results = []
        for doc, distance in combined_results:
            doc_key = doc.key if hasattr(doc, 'key') else str(doc)[:100]
            if doc_key not in seen_keys:
                seen_keys.add(doc_key)
                unique_results.append((doc, distance))

        # 上位k件に絞る
        unique_results = unique_results[:k]

        # 精製処理
        refined_docs = []
        if use_refinement and unique_results:
            for doc, distance in unique_results[:5]:  # 上位5件を精製
                doc_content = doc.text if hasattr(doc, 'text') else str(doc)
                refined = self.refiner.refine(
                    document=doc_content,
                    query=query
                )
                refined_docs.append(refined)

        confidence = self._calculate_overall_confidence(unique_results, refined_docs)

        return CRAGSearchResult(
            documents=unique_results,
            relevance_level=RelevanceLevel.AMBIGUOUS,
            strategy_used="combined_strategies",
            refined_documents=refined_docs,
            total_confidence=confidence
        )

    def _analyze_failure_patterns(
        self,
        results: List[Tuple[Any, float]]
    ) -> Dict[str, List[str]]:
        """
        失敗パターンを分析

        Args:
            results: 検索結果

        Returns:
            失敗パターン
        """
        # 簡易的な分析（実際には評価結果を使用）
        patterns = {}

        if not results:
            patterns["no_results"] = ["検索結果なし"]
        elif len(results) < 3:
            patterns["few_results"] = ["検索結果が少ない"]

        # TODO: より詳細な分析を実装
        # - 時期のミスマッチ
        # - ドキュメントタイプのミスマッチ
        # - 専門用語のミスマッチ

        return patterns

    def _select_strategies(
        self,
        failure_patterns: Dict[str, List[str]]
    ) -> List[SearchStrategy]:
        """
        失敗パターンに基づいて戦略を選択

        Args:
            failure_patterns: 失敗パターン

        Returns:
            選択された戦略リスト
        """
        selected = []

        # パターンに応じた戦略選択
        if "no_results" in failure_patterns or "few_results" in failure_patterns:
            # より広い検索戦略
            selected.append(self.search_strategies["abstract_concepts"])
            selected.append(self.search_strategies["similar_industry"])

        if "時期ミスマッチ" in str(failure_patterns):
            selected.append(self.search_strategies["temporal_expansion"])

        if "ドキュメントタイプミスマッチ" in str(failure_patterns):
            selected.append(self.search_strategies["document_type_variation"])

        # デフォルト戦略
        if not selected:
            selected = [
                self.search_strategies["detailed_same_project"],
                self.search_strategies["similar_industry"],
                self.search_strategies["same_tech_stack"]
            ]

        # 優先度でソート
        selected.sort(key=lambda x: x.priority)

        return selected[:3]  # 最大3つの戦略

    def _execute_strategy_search(
        self,
        query: str,
        project_name: str,
        k: int,
        strategy: SearchStrategy
    ) -> List[Tuple[Any, float]]:
        """
        特定の戦略で検索を実行

        Args:
            query: 検索クエリ
            project_name: プロジェクト名
            k: 取得する文書数
            strategy: 検索戦略

        Returns:
            検索結果
        """
        try:
            # メタデータフィルタに基づいて検索方法を選択
            scope = strategy.metadata_filter.get("scope", "all")

            if scope == "same_project":
                # 同一プロジェクト内検索
                return self.retriever.search_similar_documents(
                    query=query,
                    project_name=project_name,
                    k=k
                )
            elif scope == "same_industry" or scope == "same_tech":
                # 類似プロジェクト検索
                return self.retriever.get_cross_project_insights(
                    query=query,
                    exclude_project=project_name,
                    k=k
                )
            else:
                # 全体検索
                return self.retriever.search_similar_documents(
                    query=query,
                    project_name=None,  # プロジェクト制限なし
                    k=k
                )
        except Exception as e:
            print(f"[CRAG] 戦略実行エラー: {e}")
            return []

    def _abstract_query(self, query: str) -> str:
        """
        クエリを抽象化

        Args:
            query: 元のクエリ

        Returns:
            抽象化されたクエリ
        """
        # 具体的な用語を除去し、より一般的な用語に置換
        abstract_mappings = {
            "実装": "方法",
            "コード": "実現",
            "バグ": "問題",
            "エラー": "課題",
            "API": "インターフェース",
            "データベース": "データ管理"
        }

        abstract_query = query
        for specific, general in abstract_mappings.items():
            abstract_query = abstract_query.replace(specific, general)

        return abstract_query

    def _calculate_overall_confidence(
        self,
        results: List[Tuple[Any, float]],
        refined_docs: List[RefinedDocument]
    ) -> float:
        """
        全体の確信度を計算

        Args:
            results: 検索結果
            refined_docs: 精製されたドキュメント

        Returns:
            確信度 (0.0 to 1.0)
        """
        if not results:
            return 0.0

        # 検索結果の数による確信度
        result_confidence = min(1.0, len(results) / 10.0)

        # 精製の確信度の平均
        if refined_docs:
            refine_confidence = sum(d.confidence for d in refined_docs) / len(refined_docs)
        else:
            refine_confidence = 0.5

        # 総合確信度
        return result_confidence * 0.4 + refine_confidence * 0.6


def create_internal_crag(retriever, **kwargs) -> InternalCRAG:
    """
    Internal CRAGのファクトリー関数

    Args:
        retriever: RAGRetriever インスタンス
        **kwargs: その他のパラメータ

    Returns:
        InternalCRAGインスタンス
    """
    return InternalCRAG(retriever, **kwargs)