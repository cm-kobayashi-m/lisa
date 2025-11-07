"""
ドキュメント生成の思考プロセス分析機能
ヒアリングシートや提案書生成時のLLMの思考プロセスを分析・保存する
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class GeminiQuotaError(Exception):
    """Geminiのクォータ制限エラー"""
    pass


def _is_quota_error(exception: Exception) -> bool:
    """例外がクォータエラーかどうかを判定"""
    error_str = str(exception).lower()
    quota_keywords = ['quota', 'rate limit', 'resource exhausted', 'too many requests', '429']
    return any(keyword in error_str for keyword in quota_keywords)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(GeminiQuotaError)
)
def analyze_document_generation_process(
    client: genai.Client,
    document_type: str,
    source_document: str,
    generated_document: str,
    rag_results: Optional[str] = None,
    project_context: Optional[Dict[str, Any]] = None,
    additional_prompt: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    ドキュメント生成の思考プロセスを分析

    Args:
        client: Gemini APIクライアント
        document_type: ドキュメント種別（hearing-sheet, proposal）
        source_document: 入力ドキュメント（リフレクションノートやヒアリングシート）
        generated_document: 生成されたドキュメント
        rag_results: 使用されたRAG検索結果（オプション）
        project_context: プロジェクトコンテキスト
        additional_prompt: 追加指示
        model_name: 使用するモデル名

    Returns:
        思考プロセスの分析結果（JSON形式）
    """

    if model_name is None:
        # リフレクションノートと同じ環境変数を使用
        model_name = os.getenv('THOUGHT_ANALYSIS_MODEL', os.getenv('GEMINI_MODEL', 'gemini-2.5-pro '))

    # ドキュメント種別の日本語名
    doc_type_ja = {
        'hearing-sheet': 'ヒアリングシート',
        'proposal': '提案書'
    }.get(document_type, document_type)

    # コンテキストの要約
    source_summary = source_document[:3000] if len(source_document) > 3000 else source_document
    generated_summary = generated_document[:3000] if len(generated_document) > 3000 else generated_document

    rag_summary = ""
    if rag_results:
        rag_lines = rag_results.split('\n')[:30]
        rag_summary = '\n'.join(rag_lines)
        if len(rag_lines) < len(rag_results.split('\n')):
            rag_summary += "\n...(以下省略)"

    # 分析プロンプトの構築
    analysis_prompt = f"""
あなたが{doc_type_ja}を生成した際の、詳細な思考プロセスと判断根拠を教えてください。
特に「入力ドキュメントのどの部分」から「何を読み取り」「どのような変換・生成戦略」を適用したのか、具体的に説明してください。

## ドキュメント種別
{doc_type_ja}

## 入力ドキュメント（要約）
{source_summary}

## 生成されたドキュメント（要約）
{generated_summary}

## RAG検索結果（要約）
{rag_summary if rag_summary else "なし"}

## プロジェクトコンテキスト
{json.dumps(project_context, ensure_ascii=False, indent=2) if project_context else "なし"}

## 追加指示
{additional_prompt if additional_prompt else "なし"}

## 質問事項
以下の観点について、できる限り具体的かつ詳細に、JSONフォーマットで回答してください：

### ドキュメント生成特有の観点：
1. **transformation_strategy**: 入力から出力への変換戦略
   - 入力ドキュメントのどの要素をどう変換したか
   - 情報の取捨選択の基準
   - 新たに生成・推論した要素

2. **template_adaptation**: テンプレートの適応方法
   - テンプレートの各項目をどう埋めたか
   - カスタマイズした部分とその理由
   - 標準形式から逸脱した部分

3. **rag_utilization**: RAG検索結果の活用
   - どの類似案件を参考にしたか
   - 具体的にどの部分を採用/不採用としたか
   - 類似案件から学んだパターン

4. **document_specific_decisions**: ドキュメント固有の判断
   - ヒアリングシートの場合：質問の選定と順序、深掘りすべき領域の判断
   - 提案書の場合：訴求ポイントの選定、説得力を高める構成

## 出力形式
以下のJSONスキーマに従って出力してください。
**重要**: 具体的で詳細な内容を含めてください。

{{
    "document_type": "{document_type}",
    "transformation_strategy": {{
        "overall_approach": "全体的な変換アプローチ（300文字程度）",
        "key_transformations": [
            {{
                "source_element": "入力ドキュメントの該当要素",
                "transformed_to": "出力ドキュメントでの表現",
                "transformation_logic": "変換ロジックと理由"
            }}
        ],
        "information_selection": {{
            "included": ["含めた情報とその理由"],
            "excluded": ["除外した情報とその理由"],
            "added": ["新たに追加・推論した情報とその根拠"]
        }}
    }},
    "template_adaptation": {{
        "template_sections": [
            {{
                "section_name": "セクション名",
                "content_source": "内容の情報源（入力ドキュメント/RAG/推論）",
                "filling_strategy": "どのように内容を埋めたか",
                "customizations": "カスタマイズした部分"
            }}
        ],
        "deviations": [
            {{
                "standard_format": "標準的な形式",
                "actual_format": "実際に採用した形式",
                "reason": "変更理由"
            }}
        ]
    }},
    "rag_utilization": {{
        "relevant_cases": [
            {{
                "case_identifier": "類似案件の識別情報",
                "similarity_points": ["類似点"],
                "adopted_elements": ["採用した要素"],
                "rejected_elements": ["不採用とした要素とその理由"]
            }}
        ],
        "patterns_learned": ["類似案件から学んだパターン"],
        "adaptation_strategy": "類似案件の知見をどう現在の案件に適応したか"
    }},
    "document_specific_decisions": {{
        "hearing_sheet": {{
            "question_selection": [
                {{
                    "question_category": "質問カテゴリ",
                    "priority": "高/中/低",
                    "reasoning": "優先度の判断理由"
                }}
            ],
            "exploration_areas": ["深掘りすべきと判断した領域とその理由"],
            "question_flow": "質問の流れの設計意図"
        }},
        "proposal": {{
            "selling_points": [
                {{
                    "point": "訴求ポイント",
                    "evidence": "根拠となる情報源",
                    "positioning": "提案書内での位置づけ"
                }}
            ],
            "persuasion_structure": "説得力を高めるための構成戦略",
            "risk_mitigation": "リスクや懸念への対処方法"
        }}
    }},
    "reasoning_process": {{
        "main_flow": "全体的な思考の流れ（300文字程度）",
        "critical_insights": [
            {{
                "insight": "重要な洞察",
                "source": "洞察の源泉",
                "impact": "ドキュメントへの影響"
            }}
        ],
        "assumptions": ["前提とした仮定"],
        "uncertainties": ["不確実な部分とその対処"]
    }},
    "quality_assessment": {{
        "completeness": "網羅性の自己評価",
        "relevance": "関連性の自己評価",
        "clarity": "明確性の自己評価",
        "effectiveness": "効果性の自己評価",
        "improvement_areas": ["改善可能な領域"],
        "confidence_level": {{
            "overall": "high/medium/low",
            "rationale": "確信度の根拠"
        }}
    }},
    "alternative_approaches": [
        {{
            "approach": "検討した別のアプローチ",
            "pros": "利点",
            "cons": "欠点",
            "why_not_chosen": "採用しなかった理由"
        }}
    ],
    "meta_reflection": {{
        "generation_challenges": ["生成時に直面した課題"],
        "creative_decisions": ["創造的な判断を行った箇所"],
        "learning_points": ["このドキュメント生成から得た学び"],
        "reusable_patterns": ["再利用可能なパターン"]
    }}
}}

JSONのみを出力。具体的で詳細な内容を心がけてください。
"""

    try:
        # Gemini APIを呼び出し
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=analysis_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    max_output_tokens=32768,
                    temperature=0.2
                )
            )
        except Exception as e:
            print(f"[WARN] response_mime_type付きでエラー、通常モードで再試行: {e}")
            response = client.models.generate_content(
                model=model_name,
                contents=analysis_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=32768,
                    temperature=0.2
                )
            )

        # レスポンステキストを取得
        response_text = response.text.strip()

        # コードブロックを除去
        if response_text.startswith('```'):
            import re
            code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
            if code_block_match:
                response_text = code_block_match.group(1).strip()

        # JSONパース
        try:
            thought_process = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSONパースエラー: {e}")
            print(f"[DEBUG] レスポンスの最初の500文字: {response_text[:500]}")

            # JSON抽出と修復を試みる
            import re

            def extract_json_object(text):
                start = text.find('{')
                if start == -1:
                    return None

                bracket_count = 0
                in_string = False
                escape_next = False

                for i in range(start, len(text)):
                    char = text[i]

                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                return text[start:i+1]

                return text[start:]

            json_str = extract_json_object(response_text)

            if json_str:
                try:
                    thought_process = json.loads(json_str)
                except json.JSONDecodeError:
                    # 不完全なJSONを修復
                    if not json_str.rstrip().endswith('}'):
                        if '"' in json_str and json_str.count('"') % 2 == 1:
                            json_str += '"'

                        if '[' in json_str:
                            bracket_diff = json_str.count('[') - json_str.count(']')
                            json_str += ']' * bracket_diff

                        brace_diff = json_str.count('{') - json_str.count('}')
                        json_str += '}' * brace_diff

                    try:
                        thought_process = json.loads(json_str)
                    except:
                        raise ValueError(f"JSONの修復に失敗しました。レスポンスの最初の部分: {response_text[:500]}")
            else:
                raise ValueError(f"有効なJSONが見つかりません。レスポンスの最初の部分: {response_text[:500]}")

        # メタデータを追加
        thought_process['model'] = model_name
        thought_process['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        thought_process['document_type'] = document_type

        print(f"[INFO] {doc_type_ja}生成の思考プロセス分析完了")
        return thought_process

    except Exception as e:
        if _is_quota_error(e):
            raise GeminiQuotaError(str(e))
        else:
            print(f"[ERROR] 思考プロセス分析エラー: {e}")
            # エラー時はデフォルトの結果を返す
            return {
                "model": model_name,
                "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "document_type": document_type,
                "error": str(e),
                "transformation_strategy": {
                    "overall_approach": "分析に失敗しました",
                    "key_transformations": [],
                    "information_selection": {
                        "included": [],
                        "excluded": [],
                        "added": []
                    }
                },
                "quality_assessment": {
                    "completeness": "low",
                    "relevance": "low",
                    "clarity": "low",
                    "effectiveness": "low",
                    "improvement_areas": ["分析エラーが発生しました"],
                    "confidence_level": {
                        "overall": "low",
                        "rationale": "分析エラー"
                    }
                }
            }


def save_document_thought_process(
    document_type: str,
    output_dir: Path,
    thought_process: Dict[str, Any],
    document_file_path: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    ドキュメント生成の思考プロセスを保存

    Args:
        document_type: ドキュメント種別
        output_dir: 出力ディレクトリ（ドキュメントと同じ場所）
        thought_process: 思考プロセスの分析結果
        document_file_path: 対応するドキュメントファイルのパス

    Returns:
        (タイムスタンプ付きファイルパス, latestファイルパス)
    """

    # 保存先ディレクトリ作成（analysisサブディレクトリ）
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ドキュメント種別のプレフィックス
    doc_prefix = {
        'hearing-sheet': 'hearing_sheet',
        'proposal': 'proposal'
    }.get(document_type, document_type)

    # ドキュメントファイルパスを追加
    if document_file_path:
        thought_process['document_file'] = str(document_file_path)

    # タイムスタンプ付きファイルに保存
    timestamped_file = analysis_dir / f"{timestamp}_{doc_prefix}_thought_process.json"
    with open(timestamped_file, 'w', encoding='utf-8') as f:
        json.dump(thought_process, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 思考プロセスを保存: {timestamped_file}")

    # latestファイルにも保存
    latest_file = analysis_dir / f"{doc_prefix}_thought_process_latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(thought_process, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 最新の思考プロセスを保存: {latest_file}")

    return timestamped_file, latest_file


def format_document_thought_process_summary(thought_process: Dict[str, Any]) -> str:
    """
    ドキュメント生成の思考プロセスを人間が読みやすい形式にフォーマット

    Args:
        thought_process: 思考プロセスの分析結果

    Returns:
        フォーマットされた文字列
    """

    lines = []
    lines.append("# ドキュメント生成 思考プロセス分析結果\n")

    doc_type_ja = {
        'hearing-sheet': 'ヒアリングシート',
        'proposal': '提案書'
    }.get(thought_process.get('document_type', ''), thought_process.get('document_type', 'N/A'))

    lines.append(f"**ドキュメント種別**: {doc_type_ja}")
    lines.append(f"**生成日時**: {thought_process.get('generated_at', 'N/A')}")
    lines.append(f"**使用モデル**: {thought_process.get('model', 'N/A')}\n")

    if 'error' in thought_process:
        lines.append(f"⚠️ **エラー**: {thought_process['error']}\n")
        return '\n'.join(lines)

    # 変換戦略
    if 'transformation_strategy' in thought_process:
        lines.append("## 変換戦略")
        ts = thought_process['transformation_strategy']
        lines.append(f"\n### 全体的なアプローチ")
        lines.append(ts.get('overall_approach', 'N/A'))

        lines.append(f"\n### 主要な変換")
        for trans in ts.get('key_transformations', []):
            lines.append(f"\n**入力要素**: {trans.get('source_element', 'N/A')}")
            lines.append(f"**変換後**: {trans.get('transformed_to', 'N/A')}")
            lines.append(f"**変換ロジック**: {trans.get('transformation_logic', 'N/A')}")

        lines.append(f"\n### 情報の選択")
        info_sel = ts.get('information_selection', {})
        lines.append("**含めた情報:**")
        for item in info_sel.get('included', []):
            lines.append(f"- {item}")
        lines.append("\n**除外した情報:**")
        for item in info_sel.get('excluded', []):
            lines.append(f"- {item}")
        lines.append("\n**追加した情報:**")
        for item in info_sel.get('added', []):
            lines.append(f"- {item}")
        lines.append("")

    # テンプレート適応
    if 'template_adaptation' in thought_process:
        lines.append("## テンプレート適応")
        ta = thought_process['template_adaptation']

        lines.append("\n### セクション別の内容生成")
        for section in ta.get('template_sections', []):
            lines.append(f"\n**{section.get('section_name', 'N/A')}**")
            lines.append(f"- 情報源: {section.get('content_source', 'N/A')}")
            lines.append(f"- 生成戦略: {section.get('filling_strategy', 'N/A')}")
            lines.append(f"- カスタマイズ: {section.get('customizations', 'N/A')}")
        lines.append("")

    # RAG活用
    if 'rag_utilization' in thought_process:
        lines.append("## RAG検索結果の活用")
        ru = thought_process['rag_utilization']

        for case in ru.get('relevant_cases', []):
            lines.append(f"\n### {case.get('case_identifier', 'N/A')}")
            lines.append("**類似点:**")
            for point in case.get('similarity_points', []):
                lines.append(f"- {point}")
            lines.append("**採用要素:**")
            for elem in case.get('adopted_elements', []):
                lines.append(f"- {elem}")

        lines.append("\n**学習したパターン:**")
        for pattern in ru.get('patterns_learned', []):
            lines.append(f"- {pattern}")
        lines.append("")

    # ドキュメント固有の判断
    if 'document_specific_decisions' in thought_process:
        lines.append("## ドキュメント固有の判断")
        dsd = thought_process['document_specific_decisions']

        if 'hearing_sheet' in dsd and dsd['hearing_sheet']:
            hs = dsd['hearing_sheet']
            lines.append("\n### ヒアリングシート")
            lines.append("**質問の選定:**")
            for q in hs.get('question_selection', []):
                lines.append(f"- {q.get('question_category', 'N/A')} (優先度: {q.get('priority', 'N/A')})")
                lines.append(f"  理由: {q.get('reasoning', 'N/A')}")
            lines.append(f"\n**質問フロー設計**: {hs.get('question_flow', 'N/A')}")

        if 'proposal' in dsd and dsd['proposal']:
            prop = dsd['proposal']
            lines.append("\n### 提案書")
            lines.append("**訴求ポイント:**")
            for sp in prop.get('selling_points', []):
                lines.append(f"- {sp.get('point', 'N/A')}")
                lines.append(f"  根拠: {sp.get('evidence', 'N/A')}")
            lines.append(f"\n**説得構造**: {prop.get('persuasion_structure', 'N/A')}")
        lines.append("")

    # 品質評価
    if 'quality_assessment' in thought_process:
        lines.append("## 品質評価")
        qa = thought_process['quality_assessment']
        lines.append(f"- **網羅性**: {qa.get('completeness', 'N/A')}")
        lines.append(f"- **関連性**: {qa.get('relevance', 'N/A')}")
        lines.append(f"- **明確性**: {qa.get('clarity', 'N/A')}")
        lines.append(f"- **効果性**: {qa.get('effectiveness', 'N/A')}")

        cl = qa.get('confidence_level', {})
        lines.append(f"\n**全体的な確信度**: {cl.get('overall', 'N/A')}")
        lines.append(f"根拠: {cl.get('rationale', 'N/A')}")
        lines.append("")

    # メタ振り返り
    if 'meta_reflection' in thought_process:
        lines.append("## メタ振り返り")
        mr = thought_process['meta_reflection']

        lines.append("\n**生成時の課題:**")
        for challenge in mr.get('generation_challenges', []):
            lines.append(f"- {challenge}")

        lines.append("\n**創造的な判断:**")
        for decision in mr.get('creative_decisions', []):
            lines.append(f"- {decision}")

        lines.append("\n**学びのポイント:**")
        for learning in mr.get('learning_points', []):
            lines.append(f"- {learning}")

        lines.append("\n**再利用可能なパターン:**")
        for pattern in mr.get('reusable_patterns', []):
            lines.append(f"- {pattern}")

    return '\n'.join(lines)
