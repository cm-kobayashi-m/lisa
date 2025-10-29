#!/usr/bin/env python3
"""
Knowledge Refinement - CRAG の知識精製機能実装

ドキュメントを意味的なセグメントに分割し、
関連性の高い部分のみを抽出して再構成する。
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import os

from .evaluator import RelevanceEvaluator, EvaluationResult, create_evaluator


class DocumentType(Enum):
    """ドキュメントタイプの定義"""
    MEETING_MINUTES = "議事録"
    PROPOSAL = "提案書"
    DESIGN_DOC = "設計書"
    RFP = "RFP"
    REQUIREMENT = "要件定義書"
    TEST_REPORT = "テスト結果報告書"
    CONTRACT = "契約書"
    OTHER = "その他"


@dataclass
class DocumentSegment:
    """ドキュメントセグメント"""
    content: str                    # セグメントの内容
    start_pos: int                  # 元ドキュメントでの開始位置
    end_pos: int                    # 元ドキュメントでの終了位置
    segment_type: Optional[str]    # セグメントタイプ（見出し、本文等）
    metadata: Dict                  # その他のメタデータ


@dataclass
class RefinedDocument:
    """精製されたドキュメント"""
    original_doc: str               # 元のドキュメント
    refined_content: str            # 精製後のコンテンツ
    segments: List[DocumentSegment] # 選択されたセグメント
    total_segments: int             # 元の総セグメント数
    selected_segments: int          # 選択されたセグメント数
    confidence: float               # 精製の確信度


class KnowledgeRefiner:
    """
    Knowledge Refinement実行クラス

    ドキュメントを分割・評価・再構成する
    """

    def __init__(
        self,
        evaluator: Optional[RelevanceEvaluator] = None,
        max_segments: int = 5,
        segment_threshold: float = -0.5,
        min_segment_length: int = 50,
        max_segment_length: int = 1000
    ):
        """
        初期化

        Args:
            evaluator: 関連性評価器
            max_segments: 選択する最大セグメント数
            segment_threshold: セグメント選択の閾値
            min_segment_length: セグメントの最小文字数
            max_segment_length: セグメントの最大文字数
        """
        self.evaluator = evaluator or create_evaluator()
        self.max_segments = max_segments
        self.segment_threshold = segment_threshold
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length

    def refine(
        self,
        document: str,
        query: str,
        doc_type: Optional[DocumentType] = None,
        metadata: Optional[Dict] = None
    ) -> RefinedDocument:
        """
        ドキュメントを精製

        Args:
            document: 元のドキュメント
            query: 検索クエリ
            doc_type: ドキュメントタイプ
            metadata: ドキュメントのメタデータ

        Returns:
            精製されたドキュメント
        """
        # ドキュメントタイプの推定
        if doc_type is None:
            doc_type = self._infer_document_type(document, metadata)

        # セグメントに分割
        segments = self._split_document(document, doc_type)

        # 各セグメントを評価
        scored_segments = self._evaluate_segments(segments, query, metadata)

        # 上位セグメントを選択
        selected = self._select_top_segments(scored_segments)

        # 再構成
        refined_content = self._reconstruct_document(selected)

        # 確信度の計算
        confidence = self._calculate_confidence(scored_segments, selected)

        return RefinedDocument(
            original_doc=document,
            refined_content=refined_content,
            segments=[seg for seg, _ in selected],
            total_segments=len(segments),
            selected_segments=len(selected),
            confidence=confidence
        )

    def _infer_document_type(
        self,
        document: str,
        metadata: Optional[Dict]
    ) -> DocumentType:
        """
        ドキュメントタイプを推定

        Args:
            document: ドキュメント内容
            metadata: メタデータ

        Returns:
            推定されたドキュメントタイプ
        """
        # メタデータからの判定
        if metadata and 'doc_type' in metadata:
            type_str = metadata['doc_type']
            for doc_type in DocumentType:
                if doc_type.value in type_str:
                    return doc_type

        # 内容からの推定
        doc_lower = document[:1000].lower()

        if any(keyword in doc_lower for keyword in ["議事", "会議", "出席者", "決定事項"]):
            return DocumentType.MEETING_MINUTES
        elif any(keyword in doc_lower for keyword in ["提案", "ソリューション", "効果", "費用"]):
            return DocumentType.PROPOSAL
        elif any(keyword in doc_lower for keyword in ["設計", "アーキテクチャ", "実装", "仕様"]):
            return DocumentType.DESIGN_DOC
        elif any(keyword in doc_lower for keyword in ["rfp", "要求", "評価基準", "入札"]):
            return DocumentType.RFP
        elif any(keyword in doc_lower for keyword in ["要件", "機能要件", "非機能要件"]):
            return DocumentType.REQUIREMENT
        elif any(keyword in doc_lower for keyword in ["テスト", "試験", "結果", "不具合"]):
            return DocumentType.TEST_REPORT
        elif any(keyword in doc_lower for keyword in ["契約", "条項", "納期", "支払"]):
            return DocumentType.CONTRACT
        else:
            return DocumentType.OTHER

    def _split_document(
        self,
        document: str,
        doc_type: DocumentType
    ) -> List[DocumentSegment]:
        """
        ドキュメントをセグメントに分割

        Args:
            document: ドキュメント内容
            doc_type: ドキュメントタイプ

        Returns:
            セグメントのリスト
        """
        if doc_type == DocumentType.MEETING_MINUTES:
            return self._split_meeting_minutes(document)
        elif doc_type == DocumentType.PROPOSAL:
            return self._split_proposal(document)
        elif doc_type == DocumentType.DESIGN_DOC:
            return self._split_design_doc(document)
        elif doc_type == DocumentType.RFP:
            return self._split_rfp(document)
        else:
            return self._split_generic(document)

    def _split_meeting_minutes(self, document: str) -> List[DocumentSegment]:
        """議事録を分割（アジェンダ単位）"""
        segments = []

        # アジェンダパターン
        agenda_patterns = [
            r'^[0-9]+\..*$',        # 1. アジェンダ
            r'^■.*$',               # ■ アジェンダ
            r'^【.*】.*$',          # 【アジェンダ】
            r'^\*\*.*\*\*$'         # **アジェンダ**
        ]

        lines = document.split('\n')
        current_segment = []
        current_start = 0

        for i, line in enumerate(lines):
            # アジェンダの開始を検出
            is_agenda_start = any(re.match(pattern, line.strip()) for pattern in agenda_patterns)

            if is_agenda_start and current_segment:
                # 前のセグメントを保存
                content = '\n'.join(current_segment)
                if len(content) >= self.min_segment_length:
                    segments.append(DocumentSegment(
                        content=content,
                        start_pos=current_start,
                        end_pos=i,
                        segment_type="agenda_item",
                        metadata={}
                    ))
                current_segment = [line]
                current_start = i
            else:
                current_segment.append(line)

        # 最後のセグメントを保存
        if current_segment:
            content = '\n'.join(current_segment)
            if len(content) >= self.min_segment_length:
                segments.append(DocumentSegment(
                    content=content,
                    start_pos=current_start,
                    end_pos=len(lines),
                    segment_type="agenda_item",
                    metadata={}
                ))

        return segments

    def _split_proposal(self, document: str) -> List[DocumentSegment]:
        """提案書を分割（セクション単位）"""
        segments = []

        # セクションパターン
        section_patterns = [
            r'^#+\s+.*$',           # Markdown見出し
            r'^[0-9]+\..*$',        # 番号付き見出し
            r'^第[0-9一二三四五六七八九十]+[章節項].*$',  # 第N章/節/項
        ]

        lines = document.split('\n')
        current_segment = []
        current_start = 0
        current_type = "content"

        for i, line in enumerate(lines):
            is_section_start = any(re.match(pattern, line.strip()) for pattern in section_patterns)

            if is_section_start and current_segment:
                content = '\n'.join(current_segment)
                if len(content) >= self.min_segment_length:
                    segments.append(DocumentSegment(
                        content=content,
                        start_pos=current_start,
                        end_pos=i,
                        segment_type=current_type,
                        metadata={}
                    ))
                current_segment = [line]
                current_start = i
                current_type = "section"
            else:
                current_segment.append(line)

                # セグメントが長すぎる場合は分割
                if len('\n'.join(current_segment)) > self.max_segment_length:
                    content = '\n'.join(current_segment[:-1])
                    segments.append(DocumentSegment(
                        content=content,
                        start_pos=current_start,
                        end_pos=i,
                        segment_type=current_type,
                        metadata={}
                    ))
                    current_segment = [line]
                    current_start = i

        # 最後のセグメントを保存
        if current_segment:
            content = '\n'.join(current_segment)
            if len(content) >= self.min_segment_length:
                segments.append(DocumentSegment(
                    content=content,
                    start_pos=current_start,
                    end_pos=len(lines),
                    segment_type=current_type,
                    metadata={}
                ))

        return segments

    def _split_design_doc(self, document: str) -> List[DocumentSegment]:
        """設計書を分割（コンポーネント/機能単位）"""
        segments = []

        # 設計書特有のセクション
        component_keywords = [
            "コンポーネント", "モジュール", "クラス", "関数",
            "API", "インターフェース", "データベース", "テーブル"
        ]

        lines = document.split('\n')
        current_segment = []
        current_start = 0

        for i, line in enumerate(lines):
            # コンポーネントの開始を検出
            is_component_start = any(keyword in line for keyword in component_keywords)

            if is_component_start and current_segment and len(current_segment) > 3:
                content = '\n'.join(current_segment)
                if len(content) >= self.min_segment_length:
                    segments.append(DocumentSegment(
                        content=content,
                        start_pos=current_start,
                        end_pos=i,
                        segment_type="component",
                        metadata={}
                    ))
                current_segment = [line]
                current_start = i
            else:
                current_segment.append(line)

        # 最後のセグメントを保存
        if current_segment:
            content = '\n'.join(current_segment)
            if len(content) >= self.min_segment_length:
                segments.append(DocumentSegment(
                    content=content,
                    start_pos=current_start,
                    end_pos=len(lines),
                    segment_type="component",
                    metadata={}
                ))

        return segments if segments else self._split_generic(document)

    def _split_rfp(self, document: str) -> List[DocumentSegment]:
        """RFPを分割（要求項目単位）"""
        segments = []

        # 要求項目のパターン
        requirement_patterns = [
            r'^[0-9]+\.[0-9]+',    # 1.1, 1.2 形式
            r'^・',                 # 箇条書き
            r'^-\s+',              # リスト形式
            r'^\*\s+',             # アスタリスクリスト
        ]

        lines = document.split('\n')
        current_segment = []
        current_start = 0

        for i, line in enumerate(lines):
            is_requirement_start = any(re.match(pattern, line.strip()) for pattern in requirement_patterns)

            if is_requirement_start and current_segment:
                content = '\n'.join(current_segment)
                if len(content) >= self.min_segment_length:
                    segments.append(DocumentSegment(
                        content=content,
                        start_pos=current_start,
                        end_pos=i,
                        segment_type="requirement",
                        metadata={}
                    ))
                current_segment = [line]
                current_start = i
            else:
                current_segment.append(line)

        # 最後のセグメントを保存
        if current_segment:
            content = '\n'.join(current_segment)
            if len(content) >= self.min_segment_length:
                segments.append(DocumentSegment(
                    content=content,
                    start_pos=current_start,
                    end_pos=len(lines),
                    segment_type="requirement",
                    metadata={}
                ))

        return segments

    def _split_generic(self, document: str) -> List[DocumentSegment]:
        """汎用的な分割（段落単位）"""
        segments = []

        # 段落で分割
        paragraphs = re.split(r'\n\s*\n', document)

        current_pos = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) < self.min_segment_length:
                continue

            # 長すぎる段落は文単位で分割
            if len(para) > self.max_segment_length:
                sentences = re.split(r'[。！？]\s*', para)
                current_sub_segment = ""

                for sent in sentences:
                    if len(current_sub_segment) + len(sent) < self.max_segment_length:
                        current_sub_segment += sent + "。"
                    else:
                        if len(current_sub_segment) >= self.min_segment_length:
                            segments.append(DocumentSegment(
                                content=current_sub_segment,
                                start_pos=current_pos,
                                end_pos=current_pos + len(current_sub_segment),
                                segment_type="paragraph",
                                metadata={}
                            ))
                        current_sub_segment = sent + "。"
                        current_pos += len(current_sub_segment)

                if current_sub_segment and len(current_sub_segment) >= self.min_segment_length:
                    segments.append(DocumentSegment(
                        content=current_sub_segment,
                        start_pos=current_pos,
                        end_pos=current_pos + len(current_sub_segment),
                        segment_type="paragraph",
                        metadata={}
                    ))
            else:
                segments.append(DocumentSegment(
                    content=para,
                    start_pos=current_pos,
                    end_pos=current_pos + len(para),
                    segment_type="paragraph",
                    metadata={}
                ))

            current_pos += len(para) + 2  # 改行分

        return segments

    def _evaluate_segments(
        self,
        segments: List[DocumentSegment],
        query: str,
        metadata: Optional[Dict]
    ) -> List[Tuple[DocumentSegment, EvaluationResult]]:
        """
        セグメントを評価

        Args:
            segments: セグメントリスト
            query: 検索クエリ
            metadata: メタデータ

        Returns:
            (セグメント, 評価結果)のリスト
        """
        scored = []

        for segment in segments:
            result = self.evaluator.evaluate_single(
                query,
                segment.content,
                metadata
            )

            # ビジネス価値による追加評価
            business_score = self._evaluate_business_value(segment.content)

            # スコアを調整
            adjusted_score = result.score * 0.8 + business_score * 0.2
            result.score = max(-1.0, min(1.0, adjusted_score))

            scored.append((segment, result))

        return scored

    def _evaluate_business_value(self, content: str) -> float:
        """
        ビジネス価値を評価

        Args:
            content: セグメント内容

        Returns:
            ビジネス価値スコア (-1.0 to 1.0)
        """
        score = 0.0
        content_lower = content.lower()

        # ポジティブな要素
        positive_keywords = {
            "課題": 0.2,
            "問題": 0.2,
            "解決": 0.3,
            "対策": 0.3,
            "成果": 0.3,
            "効果": 0.3,
            "改善": 0.2,
            "成功": 0.3,
            "達成": 0.2,
            "利益": 0.2
        }

        for keyword, weight in positive_keywords.items():
            if keyword in content_lower:
                score += weight

        # 定量的情報の有無
        if re.search(r'\d+[%％]', content):  # パーセンテージ
            score += 0.1
        if re.search(r'\d+[万千億]?円', content):  # 金額
            score += 0.1
        if re.search(r'\d+[人名件個]', content):  # 数量
            score += 0.1

        # 最大値を1.0に制限
        return min(1.0, score)

    def _select_top_segments(
        self,
        scored_segments: List[Tuple[DocumentSegment, EvaluationResult]]
    ) -> List[Tuple[DocumentSegment, EvaluationResult]]:
        """
        上位セグメントを選択

        Args:
            scored_segments: スコア付きセグメントリスト

        Returns:
            選択されたセグメントリスト
        """
        # 閾値でフィルタリング
        filtered = [
            (seg, result) for seg, result in scored_segments
            if result.score > self.segment_threshold
        ]

        # スコアでソート
        filtered.sort(key=lambda x: x[1].score, reverse=True)

        # 最大数まで選択
        return filtered[:self.max_segments]

    def _reconstruct_document(
        self,
        selected_segments: List[Tuple[DocumentSegment, EvaluationResult]]
    ) -> str:
        """
        選択されたセグメントから文書を再構成

        Args:
            selected_segments: 選択されたセグメントリスト

        Returns:
            再構成された文書
        """
        if not selected_segments:
            return ""

        # 元の順序を保持してソート
        selected_segments.sort(key=lambda x: x[0].start_pos)

        # セグメントを結合
        parts = []
        for i, (segment, result) in enumerate(selected_segments):
            # セグメント間に区切りを追加
            if i > 0:
                parts.append("\n---\n")

            # スコアが高い場合は強調
            if result.score > 0.7:
                parts.append(f"【重要度: 高】\n{segment.content}")
            else:
                parts.append(segment.content)

        return '\n'.join(parts)

    def _calculate_confidence(
        self,
        all_segments: List[Tuple[DocumentSegment, EvaluationResult]],
        selected_segments: List[Tuple[DocumentSegment, EvaluationResult]]
    ) -> float:
        """
        精製の確信度を計算

        Args:
            all_segments: 全セグメント
            selected_segments: 選択されたセグメント

        Returns:
            確信度 (0.0 to 1.0)
        """
        if not all_segments:
            return 0.0

        # 選択率
        selection_rate = len(selected_segments) / len(all_segments)

        # 選択されたセグメントの平均スコア
        if selected_segments:
            avg_score = sum(r.score for _, r in selected_segments) / len(selected_segments)
        else:
            avg_score = 0.0

        # 確信度の計算（選択率が適度で、スコアが高い場合に高い値）
        confidence = (1.0 - abs(0.3 - selection_rate)) * avg_score

        return max(0.0, min(1.0, confidence))


def create_refiner(**kwargs) -> KnowledgeRefiner:
    """
    Knowledge Refinerのファクトリー関数

    Args:
        **kwargs: Refinerの初期化パラメータ

    Returns:
        KnowledgeRefinerインスタンス
    """
    return KnowledgeRefiner(**kwargs)