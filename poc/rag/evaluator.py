#!/usr/bin/env python3
"""
関連性評価器 - CRAG (Corrective RAG) の評価機能実装

検索結果とクエリの関連性を評価し、-1から1のスコアを返す。
社内ドキュメントの特性を考慮した評価を行う。
"""

import os
from typing import List, Dict, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai
from dataclasses import dataclass
from enum import Enum
import re


class RelevanceLevel(Enum):
    """関連性レベルの定義"""
    CORRECT = "correct"      # 高関連性（スコア >= upper_threshold）
    INCORRECT = "incorrect"  # 低関連性（スコア <= lower_threshold）
    AMBIGUOUS = "ambiguous"   # 中間（その間）


@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス"""
    score: float                    # -1.0 to 1.0
    level: RelevanceLevel           # 関連性レベル
    confidence: float               # 評価の確信度 (0.0 to 1.0)
    reasoning: str                  # 評価理由
    key_aspects: List[str]         # 関連する主要な側面


class RelevanceEvaluator:
    """
    検索結果の関連性を評価するクラス

    CRAGの評価器を社内ドキュメント向けに最適化
    """

    def __init__(
        self,
        client: genai.Client = None,
        upper_threshold: float = 0.5,
        lower_threshold: float = -0.5,
        model_name: str = None
    ):
        """
        初期化

        Args:
            client: Gemini APIクライアント
            upper_threshold: CORRECT判定の閾値
            lower_threshold: INCORRECT判定の閾値
            model_name: 使用するモデル名
        """
        self.client = client or self._initialize_client()
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.model_name = model_name or os.getenv('GEMINI_MODEL', 'gemini-2.5-pro ')

    def _initialize_client(self) -> genai.Client:
        """Gemini APIクライアントを初期化"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY が環境変数に設定されていません")
        return genai.Client(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def evaluate_single(
        self,
        query: str,
        document: str,
        document_metadata: Optional[Dict] = None
    ) -> EvaluationResult:
        """
        単一のドキュメントとクエリの関連性を評価

        Args:
            query: 検索クエリ
            document: 評価対象のドキュメント
            document_metadata: ドキュメントのメタデータ（プロジェクト名、タイプ等）

        Returns:
            評価結果
        """
        # ドキュメントの長さを制限（コンテキストウィンドウ対策）
        doc_excerpt = document[:3000] if len(document) > 3000 else document

        # メタデータ情報の構築
        metadata_info = ""
        if document_metadata:
            metadata_info = "\n【ドキュメント情報】\n"
            if 'project' in document_metadata:
                metadata_info += f"- プロジェクト: {document_metadata['project']}\n"
            if 'doc_type' in document_metadata:
                metadata_info += f"- 文書タイプ: {document_metadata['doc_type']}\n"
            if 'created_at' in document_metadata:
                metadata_info += f"- 作成日時: {document_metadata['created_at']}\n"

        prompt = f"""以下のクエリとドキュメントの関連性を評価してください。

【評価基準】
- クエリの意図とドキュメントの内容が直接的に関連しているか
- ドキュメントがクエリに対する有用な情報を含んでいるか
- ビジネスコンテキストでの価値があるか

【クエリ】
{query}
{metadata_info}

【ドキュメント内容】
{doc_excerpt}

【出力形式】
以下のJSON形式で出力してください（説明文は不要）：
{{
    "score": -1.0から1.0の数値,
    "confidence": 0.0から1.0の確信度,
    "reasoning": "評価理由（50文字以内）",
    "key_aspects": ["関連する側面1", "関連する側面2"]
}}

評価スコアの目安：
- 1.0: 完全に関連（クエリに対する直接的な回答を含む）
- 0.5: 高い関連性（有用な情報を含む）
- 0.0: 部分的な関連（間接的な情報）
- -0.5: 低い関連性（ほとんど無関係）
- -1.0: 完全に無関係
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            # レスポンスからJSONを抽出
            import json
            response_text = response.text.strip()

            # JSONブロックを抽出
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                json_str = response_text[start:end].strip()
            else:
                # 直接JSONの場合
                json_str = response_text

            result_dict = json.loads(json_str)

            # スコアに基づいてレベルを判定
            score = float(result_dict.get('score', 0))
            if score >= self.upper_threshold:
                level = RelevanceLevel.CORRECT
            elif score <= self.lower_threshold:
                level = RelevanceLevel.INCORRECT
            else:
                level = RelevanceLevel.AMBIGUOUS

            return EvaluationResult(
                score=score,
                level=level,
                confidence=float(result_dict.get('confidence', 0.5)),
                reasoning=result_dict.get('reasoning', ''),
                key_aspects=result_dict.get('key_aspects', [])
            )

        except Exception as e:
            print(f"[WARN] 評価エラー: {e}")
            # フォールバック: 簡単なキーワードマッチングで評価
            return self._fallback_evaluation(query, document)

    def _fallback_evaluation(self, query: str, document: str) -> EvaluationResult:
        """
        APIエラー時のフォールバック評価
        簡単なキーワードマッチングで評価
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', document.lower()[:1000]))

        # 共通単語の割合でスコア計算
        if len(query_words) == 0:
            score = 0.0
        else:
            common_words = query_words & doc_words
            score = (len(common_words) / len(query_words)) * 2.0 - 1.0
            score = max(-1.0, min(1.0, score))

        if score >= self.upper_threshold:
            level = RelevanceLevel.CORRECT
        elif score <= self.lower_threshold:
            level = RelevanceLevel.INCORRECT
        else:
            level = RelevanceLevel.AMBIGUOUS

        return EvaluationResult(
            score=score,
            level=level,
            confidence=0.3,  # フォールバックなので低い確信度
            reasoning="フォールバック評価（キーワードマッチング）",
            key_aspects=[]
        )

    def evaluate_batch(
        self,
        query: str,
        documents: List[Tuple[str, Optional[Dict]]]
    ) -> Tuple[List[EvaluationResult], RelevanceLevel]:
        """
        複数のドキュメントを評価し、全体の関連性レベルを判定

        Args:
            query: 検索クエリ
            documents: (ドキュメント, メタデータ)のリスト

        Returns:
            (個別の評価結果リスト, 全体の関連性レベル)
        """
        results = []

        for doc, metadata in documents:
            result = self.evaluate_single(query, doc, metadata)
            results.append(result)

        # 全体のレベルを判定（最高スコアで判定）
        if not results:
            overall_level = RelevanceLevel.INCORRECT
        else:
            max_score = max(r.score for r in results)

            if max_score >= self.upper_threshold:
                overall_level = RelevanceLevel.CORRECT
            elif all(r.score <= self.lower_threshold for r in results):
                overall_level = RelevanceLevel.INCORRECT
            else:
                overall_level = RelevanceLevel.AMBIGUOUS

        return results, overall_level

    def analyze_failure_patterns(
        self,
        failed_results: List[EvaluationResult]
    ) -> Dict[str, List[str]]:
        """
        失敗した評価結果からパターンを分析

        Args:
            failed_results: 低スコアの評価結果リスト

        Returns:
            失敗パターンの分析結果
        """
        patterns = {
            "時期ミスマッチ": [],
            "ドキュメントタイプミスマッチ": [],
            "専門用語ミスマッチ": [],
            "プロジェクト規模ミスマッチ": [],
            "業界ミスマッチ": []
        }

        for result in failed_results:
            reasoning = result.reasoning.lower()

            # パターン検出
            if any(word in reasoning for word in ["古い", "過去", "時期", "年度"]):
                patterns["時期ミスマッチ"].append(reasoning)

            if any(word in reasoning for word in ["タイプ", "種類", "形式", "文書"]):
                patterns["ドキュメントタイプミスマッチ"].append(reasoning)

            if any(word in reasoning for word in ["用語", "専門", "技術", "言葉"]):
                patterns["専門用語ミスマッチ"].append(reasoning)

            if any(word in reasoning for word in ["規模", "大小", "サイズ"]):
                patterns["プロジェクト規模ミスマッチ"].append(reasoning)

            if any(word in reasoning for word in ["業界", "分野", "ドメイン"]):
                patterns["業界ミスマッチ"].append(reasoning)

        # 空のパターンを削除
        return {k: v for k, v in patterns.items() if v}


class DocumentTypeAwareEvaluator(RelevanceEvaluator):
    """
    ドキュメントタイプを考慮した評価器

    議事録、提案書、設計書など、ドキュメントタイプごとに
    異なる評価基準を適用
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ドキュメントタイプ別の重み付け
        self.doc_type_weights = {
            "議事録": {
                "決定事項": 0.3,
                "課題": 0.2,
                "アクションアイテム": 0.2,
                "参加者発言": 0.1
            },
            "提案書": {
                "解決策": 0.3,
                "課題認識": 0.2,
                "効果・メリット": 0.2,
                "実装計画": 0.1
            },
            "設計書": {
                "アーキテクチャ": 0.3,
                "技術仕様": 0.3,
                "実装詳細": 0.2,
                "制約事項": 0.1
            },
            "RFP": {
                "要求事項": 0.4,
                "評価基準": 0.2,
                "スコープ": 0.2,
                "制約条件": 0.1
            }
        }

    def evaluate_with_type_awareness(
        self,
        query: str,
        document: str,
        doc_type: str
    ) -> EvaluationResult:
        """
        ドキュメントタイプを考慮した評価

        Args:
            query: 検索クエリ
            document: 評価対象のドキュメント
            doc_type: ドキュメントタイプ

        Returns:
            タイプを考慮した評価結果
        """
        # 基本評価を実行
        base_result = self.evaluate_single(
            query,
            document,
            {"doc_type": doc_type}
        )

        # ドキュメントタイプに基づく補正
        if doc_type in self.doc_type_weights:
            weights = self.doc_type_weights[doc_type]

            # 各側面のキーワードをチェック
            type_bonus = 0.0
            for aspect, weight in weights.items():
                if any(keyword in document[:1000] for keyword in aspect.split('・')):
                    type_bonus += weight * 0.2  # 最大0.2のボーナス

            # スコアを補正（-1.0〜1.0の範囲内）
            adjusted_score = max(-1.0, min(1.0, base_result.score + type_bonus))

            # レベルを再判定
            if adjusted_score >= self.upper_threshold:
                level = RelevanceLevel.CORRECT
            elif adjusted_score <= self.lower_threshold:
                level = RelevanceLevel.INCORRECT
            else:
                level = RelevanceLevel.AMBIGUOUS

            return EvaluationResult(
                score=adjusted_score,
                level=level,
                confidence=base_result.confidence,
                reasoning=f"{base_result.reasoning} ({doc_type}考慮)",
                key_aspects=base_result.key_aspects + [f"{doc_type}特性"]
            )

        return base_result


def create_evaluator(
    use_type_awareness: bool = True,
    **kwargs
) -> RelevanceEvaluator:
    """
    評価器のファクトリー関数

    Args:
        use_type_awareness: ドキュメントタイプ考慮を有効にするか
        **kwargs: 評価器の初期化パラメータ

    Returns:
        評価器インスタンス
    """
    if use_type_awareness:
        return DocumentTypeAwareEvaluator(**kwargs)
    else:
        return RelevanceEvaluator(**kwargs)
