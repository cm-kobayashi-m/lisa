#!/usr/bin/env python3
"""
RAG-Fusion実装モジュール

複数クエリ検索とReciprocal Rank Fusionによる高度なRAG検索を提供します。

主要機能:
1. Reciprocal Rank Fusion (RRF): 複数の検索結果を統合
2. ハイブリッドスコアリング: 類似度と時系列の重み付け
3. 複数クエリ生成: 異なる観点からのクエリ生成
4. 並行検索: 複数クエリでの並行検索とマージ
"""
import os
import math
import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai

from .vector_store import Document


class GeminiQuotaError(Exception):
    """Gemini APIのクォータ制限エラー"""
    pass


def _is_quota_error(exception: Exception) -> bool:
    """クォータエラーかどうかを判定"""
    error_msg = str(exception)
    return '429' in error_msg or 'quota' in error_msg.lower()


def reciprocal_rank_fusion(
    results_list: List[List[Tuple]],
    k: int = 60
) -> List[Tuple]:
    """
    Reciprocal Rank Fusion (RRF) で複数の検索結果をマージ

    Args:
        results_list: 複数の検索結果のリスト（各要素は[(doc, distance), ...]形式）
        k: RRFパラメータ（デフォルト: 60、論文推奨値）

    Returns:
        RRFスコアでソートされた統合結果

    アルゴリズム:
        RRFスコア(doc) = Σ(1 / (k + rank_i))
        - k: RRFパラメータ（値が大きいほど順位差を小さく扱う）
        - rank_i: クエリiでのドキュメントの順位
    """
    if not results_list:
        return []

    print(f"    [RRF] {len(results_list)}個のクエリ結果をマージ中...")

    # ドキュメントIDごとにRRFスコアを計算
    rrf_scores = {}

    for query_idx, results in enumerate(results_list):
        for rank, (doc, distance) in enumerate(results, start=1):
            # ドキュメントの一意キー（メタデータから取得）
            doc_id = doc.key if hasattr(doc, 'key') else str(hash(doc.text[:100]))

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    'doc': doc,
                    'distance': distance,  # 最初の距離を保持
                    'rrf_score': 0.0,
                    'appearances': 0,
                    'ranks': []
                }

            # RRFスコア加算
            rrf_scores[doc_id]['rrf_score'] += 1.0 / (k + rank)
            rrf_scores[doc_id]['appearances'] += 1
            rrf_scores[doc_id]['ranks'].append((query_idx, rank))

    # RRFスコアでソート
    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )

    print(f"    [RRF] {len(ranked_results)}件のユニークなドキュメントを統合")

    # (doc, distance)形式で返す
    return [(item['doc'], item['distance']) for item in ranked_results]


def normalize_score(distance: float, metric: str = 'cosine') -> float:
    """
    距離スコアを0-1の類似度に正規化

    Args:
        distance: 元の距離スコア
        metric: 距離メトリック ('cosine', 'euclidean', 'dot')

    Returns:
        0-1の類似度スコア（1に近いほど類似）
    """
    if metric == 'cosine':
        # cosine距離: 0（完全一致）〜2（完全不一致）
        # → 類似度: 1（完全一致）〜0（完全不一致）
        return max(0.0, min(1.0, 1.0 - distance / 2.0))
    elif metric == 'euclidean':
        # ユークリッド距離: 0〜∞
        # → 類似度: 1 / (1 + distance)
        return 1.0 / (1.0 + distance)
    elif metric == 'dot':
        # ドット積: -1〜1（正規化されている場合）
        # → 類似度: (dot + 1) / 2
        return (distance + 1.0) / 2.0
    else:
        # デフォルトはcosine
        return max(0.0, min(1.0, 1.0 - distance / 2.0))


def apply_hybrid_scoring(
    results: List[Tuple],
    scoring_method: str = 'hybrid',
    time_weight: float = 0.2,
    decay_days: int = 90,
    metric: str = 'cosine'
) -> List[Tuple]:
    """
    ハイブリッドスコアリング: 類似度 + 時系列重み付け

    Args:
        results: 検索結果 [(doc, distance), ...]
        scoring_method: スコアリング方式
            - 'hybrid': 類似度と時間スコアを重み付け統合（推奨）
            - 'time_decay': 類似度に時間減衰を乗算
            - 'reranking': 類似度で検索後、時間でソート
            - 'none': 類似度のみ
        time_weight: 時間スコアの重み（0.0-1.0、デフォルト: 0.2）
        decay_days: 時間減衰の半減期（日数、デフォルト: 90）
        metric: 距離メトリック

    Returns:
        スコアリング済み結果（ソート済み）

    スコア計算:
        - hybrid: final_score = (1 - w) × similarity + w × time_score
        - time_decay: final_score = similarity × time_score
        - reranking: final_score = time_score
        - none: final_score = similarity
    """
    if not results:
        return []

    print(f"    [スコアリング] 方式={scoring_method}, 時間重み={time_weight}, 半減期={decay_days}日")

    scored_results = []

    for doc, distance in results:
        # 1. 類似度スコア計算（0-1、1に近いほど類似）
        similarity = normalize_score(distance, metric)

        # 2. 時系列スコア計算（0-1、1に近いほど新しい）
        time_score = 0.5  # デフォルト値（日付不明の場合）

        if hasattr(doc, 'metadata') and doc.metadata and 'created_at' in doc.metadata:
            try:
                created_at_str = doc.metadata['created_at']
                if isinstance(created_at_str, str):
                    # ISO 8601形式をパース
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    # 経過日数を計算
                    days_old = (datetime.now() - created_at.replace(tzinfo=None)).days
                    # 指数減衰: exp(-days / decay_days)
                    time_score = math.exp(-days_old / decay_days)
            except Exception as e:
                print(f"    [WARN] 時刻解析エラー: {e}")

        # 3. 最終スコア計算
        if scoring_method == 'hybrid':
            # 重み付け統合（デフォルト: 類似度80% + 時間20%）
            final_score = (1 - time_weight) * similarity + time_weight * time_score
        elif scoring_method == 'time_decay':
            # 時間減衰を乗算
            final_score = similarity * time_score
        elif scoring_method == 'reranking':
            # 時間のみでソート
            final_score = time_score
        else:  # 'none'
            # 類似度のみ
            final_score = similarity

        scored_results.append((doc, distance, final_score))

    # 最終スコアでソート
    scored_results.sort(key=lambda x: x[2], reverse=True)

    print(f"    [スコアリング] 完了: {len(scored_results)}件")

    # (doc, distance)形式で返す（final_scoreは破棄）
    return [(doc, dist) for doc, dist, _ in scored_results]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def generate_multiple_queries(
    client: genai.Client,
    project_name: str,
    base_context: str = "",
    num_queries: int = None
) -> List[str]:
    """
    RAG-Fusion用：同じ意図を持つ複数の異なるクエリを生成

    Args:
        client: Gemini APIクライアント
        project_name: プロジェクト名
        base_context: 追加のコンテキスト情報（オプション）
        num_queries: 生成するクエリ数（Noneの場合は環境変数から取得）

    Returns:
        生成されたクエリのリスト

    生成戦略:
        - 業界用語 vs 一般用語
        - 技術スタック vs ビジネス価値
        - 課題領域 vs 解決手段
        - プロジェクト規模 vs 実装詳細
    """
    if num_queries is None:
        num_queries = int(os.getenv('RAG_FUSION_NUM_QUERIES', '3'))

    print(f"    [クエリ生成] {num_queries}個のクエリを生成中...")

    prompt = f"""プロジェクト名「{project_name}」に関連する情報を検索するため、
異なる観点から{num_queries}つの検索クエリを生成してください。

【要件】
- 同じ情報を探すが、異なる表現・観点のクエリにする
- 以下の観点で多様化する：
  1. 業界用語 vs 一般用語
  2. 技術スタック vs ビジネス価値
  3. 課題領域 vs 解決手段
  4. プロジェクト規模 vs 実装詳細

【追加コンテキスト】
{base_context if base_context else "(なし)"}

【出力形式】
各クエリを1行ずつ、以下の形式で出力してください：
1. クエリ1の内容
2. クエリ2の内容
3. クエリ3の内容

※ 説明や追加テキストは不要です。クエリのみを出力してください。
"""

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                'temperature': 0.8,  # 多様性を高める
                'top_p': 0.95,
                'max_output_tokens': 512,
            }
        )

        if _is_quota_error(Exception(str(response))):
            raise GeminiQuotaError("API quota exceeded")

        # レスポンスからクエリを抽出
        content = response.text.strip()
        queries = []

        # 行ごとに分割して処理
        for line in content.split('\n'):
            line = line.strip()
            # 番号付きリスト（1. 2. 3.）を除去
            cleaned = re.sub(r'^\d+\.\s*', '', line)
            if cleaned and len(cleaned) > 5:  # 5文字以上を有効なクエリとする
                queries.append(cleaned)

        # num_queries個に制限
        queries = queries[:num_queries]

        # 不足している場合はプロジェクト名をベースにしたクエリで補完
        while len(queries) < num_queries:
            queries.append(f"{project_name} 関連情報")

        print(f"    [クエリ生成] 完了: {len(queries)}個")
        for i, q in enumerate(queries, 1):
            print(f"      クエリ{i}: {q[:50]}...")

        return queries

    except Exception as e:
        if _is_quota_error(e):
            print(f"    [WARNING] Gemini APIクォータ制限に達しました。リトライします...")
            raise GeminiQuotaError(str(e))

        print(f"    [ERROR] クエリ生成エラー: {e}")
        # フォールバック: プロジェクト名のみのクエリを返す
        return [project_name] * num_queries


def multi_query_search(
    retriever,
    queries: List[str],
    project_name: str = None,
    k: int = 10,
    min_score: float = None,
    use_rrf: bool = True,
    rrf_k: int = None
) -> List[Tuple]:
    """
    複数クエリで並行検索し、結果をマージ

    Args:
        retriever: RAGRetrieverインスタンス
        queries: 検索クエリのリスト
        project_name: プロジェクト名（フィルタリング用、オプション）
        k: 各クエリで取得する件数
        min_score: 最小類似度スコア（オプション）
        use_rrf: RRFを使用するか（Falseの場合は単純な結合）
        rrf_k: RRFパラメータ（Noneの場合は環境変数から取得）

    Returns:
        マージされた検索結果
    """
    if not queries:
        return []

    if rrf_k is None:
        rrf_k = int(os.getenv('RAG_FUSION_RRF_K', '60'))

    print(f"    [並行検索] {len(queries)}個のクエリで検索中...")

    all_results = []

    for i, query in enumerate(queries, 1):
        print(f"    [クエリ{i}/{len(queries)}] 検索中: {query[:50]}...")

        # プロジェクト名でフィルタリング（指定されている場合）
        if project_name:
            results = retriever.search(
                query=query,
                k=k,
                filter_metadata={'project_name': project_name}
            )
        else:
            results = retriever.search(query=query, k=k)

        # 最小スコアフィルタ
        if min_score is not None:
            metric = os.getenv('VECTOR_DISTANCE_METRIC', 'cosine')
            filtered = []
            for doc, distance in results:
                similarity = normalize_score(distance, metric)
                if similarity >= min_score:
                    filtered.append((doc, distance))
            results = filtered

        print(f"      → {len(results)}件取得")
        all_results.append(results)

    # RRFまたは単純結合
    if use_rrf:
        print(f"    [RRF] 結果をマージ中（k={rrf_k}）...")
        merged = reciprocal_rank_fusion(all_results, k=rrf_k)
    else:
        print(f"    [結合] 単純結合中...")
        # 単純な結合（重複あり）
        merged = []
        for results in all_results:
            merged.extend(results)

    print(f"    [並行検索] 完了: {len(merged)}件")
    return merged


def rag_fusion_search(
    client: genai.Client,
    retriever,
    project_name: str,
    base_query: str = None,
    k: int = 10,
    num_queries: int = None,
    min_score: float = None,
    apply_time_weighting: bool = True
) -> List[Tuple]:
    """
    RAG-Fusion統合検索フロー

    ワンストップでRAG-Fusion検索を実行：
    1. 複数のクエリを生成
    2. 各クエリで並行検索
    3. RRFでマージ
    4. ハイブリッドスコアリング適用

    Args:
        client: Gemini APIクライアント
        retriever: RAGRetrieverインスタンス
        project_name: プロジェクト名
        base_query: ベースとなる検索クエリ（オプション）
        k: 最終的に取得する件数
        num_queries: 生成するクエリ数（Noneの場合は環境変数から）
        min_score: 最小類似度スコア（オプション）
        apply_time_weighting: ハイブリッドスコアリングを適用するか

    Returns:
        RAG-Fusionで検索された結果
    """
    print(f"  [RAG-Fusion] プロジェクト「{project_name}」で検索開始")

    # 1. 複数クエリ生成
    queries = generate_multiple_queries(
        client,
        project_name,
        base_context=base_query,
        num_queries=num_queries
    )

    # 2. 並行検索 + RRFマージ
    merged_results = multi_query_search(
        retriever,
        queries,
        project_name=project_name,
        k=k * 3,  # 後でフィルタリングするため多めに取得
        min_score=min_score,
        use_rrf=True
    )

    # 3. ハイブリッドスコアリング
    if apply_time_weighting:
        print(f"  [RAG-Fusion] ハイブリッドスコアリング適用中...")
        scoring_method = os.getenv('RAG_SCORING_METHOD', 'hybrid')
        time_weight = float(os.getenv('RAG_TIME_WEIGHT', '0.2'))
        decay_days = int(os.getenv('RAG_DECAY_DAYS', '90'))

        scored_results = apply_hybrid_scoring(
            merged_results,
            scoring_method=scoring_method,
            time_weight=time_weight,
            decay_days=decay_days
        )
    else:
        scored_results = merged_results

    # 4. 最終的にk件に絞る
    final_results = scored_results[:k]

    print(f"  [RAG-Fusion] 検索完了: 最終結果 {len(final_results)}件")
    return final_results
