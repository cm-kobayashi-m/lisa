#!/usr/bin/env python3
"""
Query Translation for RAG Search
ユーザーの追加プロンプトと元のドキュメントから最適化された検索クエリを生成

追加プロンプトから以下を自動的に抽出：
- 参考プロジェクト名（例：「ヤーマン案件を参考に」→ ヤーマン）
- 優先事項（例：「精度重視」「スピード重視」）
- 検索戦略（例：過去案件重視、最新技術重視）
"""

from typing import List, Dict, Optional, Any
from google import genai
import os
import json
import re


def translate_query_with_context(
    client: genai.Client,
    source_document: str,
    additional_prompt: Optional[str] = None,
    num_queries: int = 3
) -> Dict[str, Any]:
    """
    追加コンテキストを考慮した検索クエリを生成

    Args:
        client: Gemini APIクライアント
        source_document: 元のソースドキュメント（リフレクションノート/ヒアリングシート）
        additional_prompt: ユーザーからの追加指示
        num_queries: 生成するクエリ数

    Returns:
        {
            "primary_query": "主要な検索クエリ",
            "alternative_queries": ["代替クエリ1", "代替クエリ2"],
            "reference_projects": ["参考プロジェクト1", "参考プロジェクト2"],
            "search_strategy": "検索戦略の説明",
            "priority": "優先事項",
            "filters": {
                "project_names": ["プロジェクト名"],
                "keywords": ["キーワード1", "キーワード2"],
                "document_types": ["ドキュメントタイプ"]
            }
        }
    """

    # 追加プロンプトがない場合はデフォルトの処理
    if not additional_prompt:
        return {
            "primary_query": source_document,
            "alternative_queries": [source_document],
            "reference_projects": [],
            "search_strategy": "標準的な検索戦略",
            "priority": "バランス重視",
            "filters": {
                "project_names": [],
                "keywords": [],
                "document_types": []
            }
        }

    # プロンプト構築
    prompt = f"""# タスク
以下の追加指示と元のドキュメントから、RAG検索で最も効果的な検索クエリと検索戦略を生成してください。

## 元のドキュメント（冒頭部分）
{source_document}

## ユーザーからの追加指示
{additional_prompt}

## 解析項目

1. **参考プロジェクトの抽出**
   - 「〜プロジェクト」「〜案件」「〜の事例」などの表現から抽出
   - 例：「ヤーマン案件を参考に」→ "ヤーマン"

2. **優先事項の判定**
   - 精度重視：「精度の高い」「正確な」「詳細な」
   - スピード重視：「期限が厳しい」「早急に」「短納期」
   - コスト重視：「予算が限られる」「コスト削減」「効率的」
   - 実績重視：「確実な」「実績のある」「成功事例」
   - 最新技術：「最新の」「トレンド」「先進的」

3. **検索戦略の決定**
   - 参考プロジェクトがある場合：類似案件を優先的に検索
   - 期限が厳しい場合：実績のある確実な手法を検索
   - 精度重視の場合：詳細な技術文書や設計書を検索
   - 最新技術の場合：最近のプロジェクトや技術トレンドを検索

4. **検索クエリの生成**
   - 元のドキュメントの主要キーワードを抽出
   - 追加指示の文脈を考慮してクエリを調整
   - {num_queries}個の異なる観点からのクエリを生成

## 出力形式（JSON）
{{
  "primary_query": "最も重要な検索クエリ（100文字以内）",
  "alternative_queries": ["代替クエリ1", "代替クエリ2"],
  "reference_projects": ["参考プロジェクト名1", "参考プロジェクト名2"],
  "search_strategy": "検索戦略の簡潔な説明（100文字以内）",
  "priority": "優先事項（精度重視/スピード重視/コスト重視/実績重視/最新技術/バランス）",
  "filters": {{
    "project_names": ["フィルタに使用するプロジェクト名"],
    "keywords": ["重要キーワード1", "重要キーワード2", "重要キーワード3"],
    "document_types": ["優先するドキュメントタイプ（ヒアリングシート/提案書/設計書など）"]
  }}
}}

※ JSONのみを出力してください（コードブロックや説明文は不要）"""

    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                'temperature': 0.3,  # 安定した出力のため低め
                'max_output_tokens': 8192,
            }
        )

        # JSON解析
        if not response or not response.text:
            raise ValueError("Gemini APIから有効なレスポンスが返されませんでした")
        content = response.text.strip()

        # コードブロック除去
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)

        # ログ出力
        print(f"[Query Translation] 検索戦略: {result.get('search_strategy', '')}")
        print(f"[Query Translation] 優先事項: {result.get('priority', '')}")
        if result.get('reference_projects'):
            print(f"[Query Translation] 参考プロジェクト: {', '.join(result['reference_projects'])}")
        print(f"[Query Translation] 主要クエリ: {result.get('primary_query', '')[:50]}...")

        return result

    except Exception as e:
        print(f"[WARN] Query Translation エラー: {e}")

        # フォールバック処理：簡易的な抽出を試みる
        reference_projects = extract_project_names(additional_prompt)
        priority = detect_priority(additional_prompt)

        return {
            "primary_query": source_document,
            "alternative_queries": [source_document],
            "reference_projects": reference_projects,
            "search_strategy": f"フォールバック戦略（{priority}）",
            "priority": priority,
            "filters": {
                "project_names": reference_projects,
                "keywords": [],
                "document_types": []
            }
        }


def extract_project_names(text: Optional[str]) -> List[str]:
    """
    テキストからプロジェクト名を簡易的に抽出（フォールバック用）

    Args:
        text: 解析対象テキスト

    Returns:
        抽出されたプロジェクト名のリスト
    """
    if not text:
        return []

    projects = []

    # パターンマッチングで抽出
    patterns = [
        r'([^\s]+)プロジェクト',
        r'([^\s]+)案件',
        r'([^\s]+)の事例',
        r'([^\s]+)を参考',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        projects.extend(matches)

    # 重複除去
    return list(set(projects))


def detect_priority(text: Optional[str]) -> str:
    """
    テキストから優先事項を簡易的に検出（フォールバック用）

    Args:
        text: 解析対象テキスト

    Returns:
        優先事項の文字列
    """
    if not text:
        return "バランス"

    # キーワードベースの判定
    if any(word in text for word in ["精度", "正確", "詳細", "高品質"]):
        return "精度重視"
    elif any(word in text for word in ["期限", "早急", "短納期", "スピード"]):
        return "スピード重視"
    elif any(word in text for word in ["予算", "コスト", "効率", "安価"]):
        return "コスト重視"
    elif any(word in text for word in ["確実", "実績", "成功", "安定"]):
        return "実績重視"
    elif any(word in text for word in ["最新", "トレンド", "先進", "新技術"]):
        return "最新技術"
    else:
        return "バランス"


def apply_query_filters(
    query_result: Dict[str, Any],
    project_name: str = ""
) -> Dict[str, Any]:
    """
    Query Translation結果にプロジェクトフィルタを適用

    Args:
        query_result: translate_query_with_contextの結果
        project_name: 現在のプロジェクト名（RAG検索用）

    Returns:
        フィルタが適用された検索パラメータ
    """
    # 参考プロジェクトがある場合は優先的に検索
    if query_result.get("reference_projects"):
        # 参考プロジェクト名を検索対象に追加
        search_projects = query_result["reference_projects"]

        # 現在のプロジェクト名も追加（フォールバック用）
        if project_name and project_name not in search_projects:
            search_projects.append(project_name)
    else:
        # 参考プロジェクトがない場合は現在のプロジェクトのみ
        search_projects = [project_name] if project_name else []

    # 検索パラメータを構築
    search_params = {
        "queries": [query_result["primary_query"]] + query_result.get("alternative_queries", []),
        "project_filter": search_projects,
        "keywords": query_result.get("filters", {}).get("keywords", []),
        "document_types": query_result.get("filters", {}).get("document_types", []),
        "strategy": query_result.get("search_strategy", "標準検索"),
        "priority": query_result.get("priority", "バランス")
    }

    return search_params


if __name__ == "__main__":
    # テスト実行
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    # テスト用のクライアント初期化
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY が設定されていません")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # テストケース
    test_cases = [
        {
            "document": "ECサイト構築プロジェクトのリフレクションノート。主な要件：商品管理、決済システム、在庫管理...",
            "prompt": "ヤーマンプロジェクトを参考にして、期限が厳しいので確実な手法でお願いします"
        },
        {
            "document": "AI画像認識システムの提案書。機械学習モデルの開発と実装...",
            "prompt": "最新の技術トレンドを取り入れた提案にしてください。過去のAI案件も参考に"
        },
        {
            "document": "基幹システム刷新プロジェクト。レガシーシステムからの移行...",
            "prompt": "予算が限られているので、コスト効率の良い方法で"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"テストケース {i}")
        print(f"{'='*60}")
        print(f"追加プロンプト: {test['prompt']}")
        print(f"\n結果:")

        result = translate_query_with_context(
            client=client,
            source_document=test["document"],
            additional_prompt=test["prompt"]
        )

        print(json.dumps(result, ensure_ascii=False, indent=2))
