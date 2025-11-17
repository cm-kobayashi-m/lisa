"""
思考プロセス分析機能
リフレクションノート生成後、LLMの思考プロセスを分析・保存する
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 共通ユーティリティ
from utils.gemini_utils import (
    GeminiQuotaError,
    is_quota_error,
    extract_json_from_response
)
from utils.thought_utils import (
    build_yaml_frontmatter,
    enhance_markdown_with_metadata
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def analyze_thought_process(
    client: genai.Client,
    project_name: str,
    reflection_note: str,
    rag_context: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    リフレクションノート生成の思考プロセスを分析

    Args:
        client: Gemini APIクライアント
        project_name: プロジェクト名
        reflection_note: 生成されたリフレクションノート
        rag_context: 使用されたRAGコンテキスト（オプション）
        model_name: 使用するモデル名（デフォルト: 環境変数から取得）

    Returns:
        思考プロセスの分析結果（JSON形式）
    """

    if model_name is None:
        # 思考プロセス分析用のモデル（安価なモデルを使用可能）
        model_name = os.getenv('THOUGHT_ANALYSIS_MODEL', os.getenv('GEMINI_MODEL', 'gemini-2.5-pro '))

    # RAGコンテキストは思考プロセス分析に重要なので、より多くの情報を含める
    context_summary = ""
    if rag_context:
        # 最大80000文字まで含める（ノート生成時と同じ制限）
        if len(rag_context) > 80000:
            context_summary = rag_context[:80000] + "\n...(以下省略)"
        else:
            context_summary = rag_context

    # 分析プロンプトの構築
    analysis_prompt = f"""
あなたが以下のリフレクションノートを生成した際の、詳細な思考プロセスと判断根拠を教えてください。
特に「どの情報源のどの部分」から「何を読み取り」「どのような判断」をしたのか、具体的に説明してください。

## 生成されたリフレクションノート
{reflection_note}

## 使用されたRAGコンテキスト
以下は、ノート生成時に参照可能だったRAGドキュメントの情報です。
これらの中から実際にどのドキュメントのどの部分を参考にしたかを特定してください。

{context_summary if context_summary else "なし"}

## 質問事項
以下の観点について、できる限り具体的かつ詳細に、JSONフォーマットで回答してください：

1. **reasoning_process**: 思考の流れと根拠
   - どのドキュメント（RAG）を参考にしたか（ドキュメント名、セクション名など）
   - そのドキュメントのどの部分が重要だったか（具体的な引用や要約）
   - そこから何を読み取ったか
   - どのような推論をしたか

2. **key_decisions**: 重要な意思決定と選択理由
   - 複数の選択肢があった箇所
   - 選択した案とその決め手
   - 却下した案とその理由

3. **information_sources**: 情報源の活用方法 ⭐重要⭐
   - 上記「使用されたRAGコンテキスト」セクションに記載されている各ドキュメントについて：
     * ドキュメントの識別情報（タイトル、ファイル名など）を明確に記載
     * どのセクション・どの部分を参照したか
     * その部分の重要度（high/medium/low）と判断理由
     * ノートのどこに反映したか
   - 各情報源の信頼性評価
   - **important_sections配列には、参照した全ての重要セクションを必ず含めてください**

4. **section_composition**: セクション構成の詳細な理由
   - 各セクションを含めた具体的理由
   - セクション間の関連性の考慮
   - 読者の理解フローの設計意図

5. **analytical_depth**: 分析の深さと範囲
   - 深く掘り下げた領域とその理由
   - あえて簡潔にした領域とその判断根拠
   - 分析のスコープ決定の基準

## 出力形式
以下のJSONスキーマに従って出力してください。
**重要**: 具体的で詳細な内容を含めてください。抽象的な表現は避けてください。

{{
    "reasoning_process": {{
        "main_flow": "全体的な思考の流れ（300文字程度）",
        "key_insights": [
            {{
                "source": "参照した情報源や文書の該当箇所",
                "finding": "そこから読み取った内容",
                "reasoning": "それをどう解釈し、どう活用したか"
            }}
        ],
        "logical_connections": "各要素をどのように結びつけて結論に至ったか"
    }},
    "key_decisions": [
        {{
            "decision_point": "意思決定が必要だったポイント",
            "options_considered": [
                {{
                    "option": "検討した選択肢",
                    "pros": "利点",
                    "cons": "欠点"
                }}
            ],
            "chosen_option": "選択した案",
            "decisive_factors": ["決め手となった要因1", "決め手となった要因2"],
            "rejected_reasons": "他の案を却下した具体的理由"
        }}
    ],
    "information_sources": [
        {{
            "source_type": "情報源の種類（RAG文書、スクリーンショット、etc）",
            "document_identifier": "RAGドキュメントの場合は識別情報（ファイル名、タイトルなど）",
            "important_sections": [
                {{
                    "section_reference": "セクション名、ページ番号、または位置情報",
                    "content_summary": "重要だった部分の要約または引用",
                    "importance_level": "high/medium/low"
                }}
            ],
            "specific_content": "参照した具体的な内容",
            "interpretation": "その情報をどう解釈したか",
            "reliability_assessment": "信頼性の評価とその根拠",
            "usage_in_note": "ノートのどの部分に反映したか"
        }}
    ],
    "section_composition": {{
        "overall_structure": "全体構成の設計思想",
        "sections": [
            {{
                "name": "セクション名",
                "purpose": "このセクションの目的",
                "content_selection": "含めた内容の選定理由",
                "placement_reasoning": "この位置に配置した理由",
                "dependencies": "他セクションとの関連性"
            }}
        ],
        "flow_design": "読者の理解の流れをどう設計したか"
    }},
    "analytical_depth": {{
        "deep_dive_areas": [
            {{
                "area": "深く分析した領域",
                "reason": "深掘りした理由",
                "methodology": "分析手法",
                "findings": "得られた洞察"
            }}
        ],
        "surface_level_areas": [
            {{
                "area": "簡潔に扱った領域",
                "reason": "簡潔にした理由",
                "trade_off": "何を優先して何を省略したか"
            }}
        ],
        "scope_boundaries": "分析範囲の境界線をどこに引いたか、その理由"
    }},
    "quality_considerations": {{
        "completeness": "網羅性についての自己評価",
        "accuracy": "正確性についての自己評価",
        "usefulness": "実用性についての自己評価",
        "improvements_needed": ["認識している改善点"],
        "confidence_levels": {{
            "high_confidence_areas": ["高い確信を持っている部分とその根拠"],
            "medium_confidence_areas": ["ある程度の確信を持っている部分とその理由"],
            "low_confidence_areas": ["確信が低い部分とその理由"]
        }}
    }},
    "alternative_approaches": [
        {{
            "approach": "検討した別のアプローチ",
            "why_not_chosen": "採用しなかった理由",
            "potential_benefits": "そのアプローチの潜在的な利点",
            "conditions_for_use": "どのような条件なら採用したか"
        }}
    ],
    "meta_reflection": {{
        "thought_process_summary": "自分の思考プロセス全体への振り返り",
        "biases_acknowledged": ["認識しているバイアスや前提"],
        "uncertainties": ["不確実な部分とその対処"],
        "learning_points": ["このノート生成から得た学び"]
    }}
}}

JSONのみを出力。具体的で詳細な内容を心がけてください。
"""

    try:
        # Gemini APIを呼び出し
        # response_mime_typeがエラーの原因の可能性があるため、一時的に除外
        try:
            # まず response_mime_type 付きで試す
            response = client.models.generate_content(
                model=model_name,
                contents=analysis_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',  # JSON形式を強制
                    max_output_tokens=32768,  # トークン数を増やして途中切れを防ぐ
                    temperature=0.2  # 低温度で安定した出力
                )
            )
        except Exception as e:
            print(f"[WARN] response_mime_type付きでエラー、通常モードで再試行: {e}")
            # response_mime_type なしで再試行
            response = client.models.generate_content(
                model=model_name,
                contents=analysis_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=32768,
                    temperature=0.2
                )
            )

        # レスポンステキストを取得してJSONを抽出
        response_text = response.text.strip()
        thought_process = extract_json_from_response(response_text)

        if thought_process is None:
            raise ValueError(f"JSONの抽出に失敗しました。レスポンスの最初の部分: {response_text[:500]}")

        # メタデータを追加
        thought_process['model'] = model_name
        thought_process['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        thought_process['project_name'] = project_name

        print(f"[INFO] 思考プロセス分析完了")
        return thought_process

    except Exception as e:
        if is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[ERROR] 思考プロセス分析エラー: {e}")
            # エラー時はデフォルトの結果を返す
            return {
                "model": model_name,
                "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "project_name": project_name,
                "error": str(e),
                "reasoning_summary": "分析に失敗しました",
                "decision_criteria": [],
                "section_order_rationale": [],
                "included_items": [],
                "excluded_items": [],
                "comparative_considerations": [],
                "improvement_selection_criteria": [],
                "tradeoffs": [],
                "limitations": ["分析エラーが発生しました"],
                "confidence": {
                    "overall": "low",
                    "section_structure": "low",
                    "improvements": "low",
                    "comparative_analysis": "low"
                }
            }




def save_thought_process(
    project_name: str,
    thought_process: Dict[str, Any],
    note_file_path: Optional[str] = None,
    output_dir: str = "outputs",
    save_markdown: bool = True
) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    """
    思考プロセスを保存

    Args:
        project_name: プロジェクト名
        thought_process: 思考プロセスの分析結果
        note_file_path: 対応するノートファイルのパス
        output_dir: 出力ディレクトリ
        save_markdown: Markdown形式でも保存するか（デフォルト: True）

    Returns:
        (JSONタイムスタンプファイル, JSONラテストファイル,
         MDタイムスタンプファイル, MDラテストファイル)
        Markdown保存が無効の場合、後者2つはNone
    """

    # 保存先ディレクトリ作成
    analysis_dir = Path(output_dir) / project_name / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ノートファイルパスを追加
    if note_file_path:
        thought_process['note_file'] = str(note_file_path)

    # タイムスタンプ付きファイルに保存
    timestamped_file = analysis_dir / f"{timestamp}_note_thought_process.json"
    with open(timestamped_file, 'w', encoding='utf-8') as f:
        json.dump(thought_process, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 思考プロセスを保存: {timestamped_file}")

    # latestファイルにも保存
    latest_file = analysis_dir / "note_thought_process_latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(thought_process, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 最新の思考プロセスを保存: {latest_file}")

    # Markdown形式でも保存
    md_timestamped_file = None
    md_latest_file = None

    if save_markdown:
        try:
            # format_thought_process_summary()を使ってMarkdown本文を生成
            markdown_body = format_thought_process_summary(thought_process)

            # ノート固有の目次セクション
            toc_sections = [
                "思考の流れと根拠",
                "重要な意思決定",
                "情報源の活用",
                "セクション構成",
                "分析の深さ",
                "品質評価",
                "改善の余地"
            ]

            # メタデータを追加（YAMLフロントマター、目次）
            # プロジェクト名を基準とした相対パスを計算
            json_relative_path = f"{project_name}/analysis/{timestamped_file.name}"
            enhanced_markdown = enhance_markdown_with_metadata(
                markdown_body,
                thought_process,
                json_file_path=json_relative_path,
                toc_sections=toc_sections
            )

            # タイムスタンプ付きMarkdownファイルに保存
            md_timestamped_file = analysis_dir / f"{timestamp}_note_thought_process.md"
            with open(md_timestamped_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_markdown)
            print(f"[INFO] 思考プロセス（Markdown）を保存: {md_timestamped_file}")

            # latestファイルにも保存
            md_latest_file = analysis_dir / "note_thought_process_latest.md"
            with open(md_latest_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_markdown)
            print(f"[INFO] 最新の思考プロセス（Markdown）を保存: {md_latest_file}")

        except Exception as e:
            print(f"[ERROR] Markdown保存エラー: {e}")
            # Markdownの保存に失敗してもJSONは保存されているので処理は継続

    return timestamped_file, latest_file, md_timestamped_file, md_latest_file


def load_thought_process(file_path: Path) -> Dict[str, Any]:
    """
    保存された思考プロセスを読み込む

    Args:
        file_path: JSONファイルのパス

    Returns:
        思考プロセスの辞書
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_thought_process_summary(thought_process: Dict[str, Any]) -> str:
    """
    思考プロセスを人間が読みやすい形式にフォーマット

    Args:
        thought_process: 思考プロセスの分析結果

    Returns:
        フォーマットされた文字列
    """

    lines = []
    lines.append("# 思考プロセス分析結果\n")
    lines.append(f"**プロジェクト**: {thought_process.get('project_name', 'N/A')}")
    lines.append(f"**生成日時**: {thought_process.get('generated_at', 'N/A')}")
    lines.append(f"**使用モデル**: {thought_process.get('model', 'N/A')}\n")

    if 'error' in thought_process:
        lines.append(f"⚠️ **エラー**: {thought_process['error']}\n")
        return '\n'.join(lines)

    # 思考プロセスの流れ
    if 'reasoning_process' in thought_process:
        lines.append("## 思考プロセスの流れ")
        rp = thought_process['reasoning_process']
        lines.append(f"\n### 全体的な思考の流れ")
        lines.append(rp.get('main_flow', 'N/A'))

        lines.append(f"\n### 重要な洞察")
        for insight in rp.get('key_insights', []):
            lines.append(f"\n**情報源**: {insight.get('source', 'N/A')}")
            lines.append(f"**発見**: {insight.get('finding', 'N/A')}")
            lines.append(f"**推論**: {insight.get('reasoning', 'N/A')}")

        lines.append(f"\n### 論理的な結びつけ")
        lines.append(rp.get('logical_connections', 'N/A'))
        lines.append("")

    # 重要な意思決定
    if 'key_decisions' in thought_process:
        lines.append("## 重要な意思決定")
        for decision in thought_process['key_decisions']:
            lines.append(f"\n### {decision.get('decision_point', 'N/A')}")

            lines.append("\n**検討した選択肢:**")
            for opt in decision.get('options_considered', []):
                lines.append(f"- **{opt.get('option', 'N/A')}**")
                lines.append(f"  - 利点: {opt.get('pros', 'N/A')}")
                lines.append(f"  - 欠点: {opt.get('cons', 'N/A')}")

            lines.append(f"\n**選択した案**: {decision.get('chosen_option', 'N/A')}")
            lines.append(f"**決め手となった要因**:")
            for factor in decision.get('decisive_factors', []):
                lines.append(f"- {factor}")
            lines.append(f"**却下理由**: {decision.get('rejected_reasons', 'N/A')}")
        lines.append("")

    # 情報源の活用
    if 'information_sources' in thought_process:
        lines.append("## 情報源の活用")
        for source in thought_process['information_sources']:
            source_title = source.get('document_identifier') or source.get('source_type', 'N/A')
            lines.append(f"\n### {source_title}")
            lines.append(f"**種類**: {source.get('source_type', 'N/A')}")

            # 重要なセクションの詳細表示
            if 'important_sections' in source and source['important_sections']:
                lines.append("\n**重要なセクション:**")
                for section in source.get('important_sections', []):
                    lines.append(f"\n- **{section.get('section_reference', 'N/A')}** (重要度: {section.get('importance_level', 'N/A')})")
                    lines.append(f"  - 内容: {section.get('content_summary', 'N/A')}")

            lines.append(f"\n**具体的な内容**: {source.get('specific_content', 'N/A')}")
            lines.append(f"**解釈**: {source.get('interpretation', 'N/A')}")
            lines.append(f"**信頼性評価**: {source.get('reliability_assessment', 'N/A')}")
            lines.append(f"**ノートへの反映**: {source.get('usage_in_note', 'N/A')}")
        lines.append("")

    # セクション構成
    if 'section_composition' in thought_process:
        lines.append("## セクション構成の詳細")
        sc = thought_process['section_composition']
        lines.append(f"\n### 全体構成の設計思想")
        lines.append(sc.get('overall_structure', 'N/A'))

        lines.append(f"\n### 各セクションの詳細")
        for section in sc.get('sections', []):
            lines.append(f"\n**{section.get('name', 'N/A')}**")
            lines.append(f"- 目的: {section.get('purpose', 'N/A')}")
            lines.append(f"- 内容選定理由: {section.get('content_selection', 'N/A')}")
            lines.append(f"- 配置理由: {section.get('placement_reasoning', 'N/A')}")
            lines.append(f"- 他セクションとの関連: {section.get('dependencies', 'N/A')}")

        lines.append(f"\n### 読者の理解フロー設計")
        lines.append(sc.get('flow_design', 'N/A'))
        lines.append("")

    # 分析の深さ
    if 'analytical_depth' in thought_process:
        lines.append("## 分析の深さと範囲")
        ad = thought_process['analytical_depth']

        lines.append("\n### 深く分析した領域")
        for area in ad.get('deep_dive_areas', []):
            lines.append(f"\n**{area.get('area', 'N/A')}**")
            lines.append(f"- 理由: {area.get('reason', 'N/A')}")
            lines.append(f"- 手法: {area.get('methodology', 'N/A')}")
            lines.append(f"- 洞察: {area.get('findings', 'N/A')}")

        lines.append("\n### 簡潔に扱った領域")
        for area in ad.get('surface_level_areas', []):
            lines.append(f"\n**{area.get('area', 'N/A')}**")
            lines.append(f"- 理由: {area.get('reason', 'N/A')}")
            lines.append(f"- トレードオフ: {area.get('trade_off', 'N/A')}")

        lines.append(f"\n### スコープの境界")
        lines.append(ad.get('scope_boundaries', 'N/A'))
        lines.append("")

    # 品質考慮事項
    if 'quality_considerations' in thought_process:
        lines.append("## 品質に関する考慮事項")
        qc = thought_process['quality_considerations']
        lines.append(f"- **網羅性**: {qc.get('completeness', 'N/A')}")
        lines.append(f"- **正確性**: {qc.get('accuracy', 'N/A')}")
        lines.append(f"- **実用性**: {qc.get('usefulness', 'N/A')}")

        lines.append("\n### 改善が必要な点")
        for improvement in qc.get('improvements_needed', []):
            lines.append(f"- {improvement}")

        lines.append("\n### 確信度レベル")
        cl = qc.get('confidence_levels', {})
        lines.append("**高い確信度:**")
        for area in cl.get('high_confidence_areas', []):
            lines.append(f"- {area}")
        lines.append("\n**中程度の確信度:**")
        for area in cl.get('medium_confidence_areas', []):
            lines.append(f"- {area}")
        lines.append("\n**低い確信度:**")
        for area in cl.get('low_confidence_areas', []):
            lines.append(f"- {area}")
        lines.append("")

    # 代替アプローチ
    if 'alternative_approaches' in thought_process:
        lines.append("## 検討した代替アプローチ")
        for alt in thought_process['alternative_approaches']:
            lines.append(f"\n### {alt.get('approach', 'N/A')}")
            lines.append(f"- 採用しなかった理由: {alt.get('why_not_chosen', 'N/A')}")
            lines.append(f"- 潜在的な利点: {alt.get('potential_benefits', 'N/A')}")
            lines.append(f"- 採用条件: {alt.get('conditions_for_use', 'N/A')}")
        lines.append("")

    # メタ振り返り
    if 'meta_reflection' in thought_process:
        lines.append("## メタ振り返り")
        mr = thought_process['meta_reflection']
        lines.append(f"\n### 思考プロセスの総括")
        lines.append(mr.get('thought_process_summary', 'N/A'))

        lines.append(f"\n### 認識しているバイアス")
        for bias in mr.get('biases_acknowledged', []):
            lines.append(f"- {bias}")

        lines.append(f"\n### 不確実性")
        for uncertainty in mr.get('uncertainties', []):
            lines.append(f"- {uncertainty}")

        lines.append(f"\n### 学びのポイント")
        for learning in mr.get('learning_points', []):
            lines.append(f"- {learning}")

    return '\n'.join(lines)
